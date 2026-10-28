"""Standalone XCOP Modifier with the Amyloid Modifier 2.2.2 engine.

Structure I/O and geometry are kept in sync with the 2.2.2 bundle. The cluster
preview and file workflow remain standalone; no ChimeraX package is required.
The optional secondary-structure button retains the existing external Chimera
workflow. Expansion itself does not invoke Chimera or ChimeraX.
"""
import sys
import csv
import shutil
import atexit
import argparse

import os
import math
from io import StringIO
from copy import deepcopy
from html import escape

import json
import tempfile
import hashlib
import base64
from itertools import product
from typing import NamedTuple
from collections import Counter, defaultdict


VERSION = '2.2.2'
# Below this per-cycle angle, a free rotation axis is dominated by coordinate
# rounding/noise. Use the measured repeat direction to constrain the rotation.
MIN_RESOLVED_ROTATION_DEGREES = 0.05
MAX_REPEAT_CORE_RMSD = 1.5
MIN_NEW_CA_SEPARATION = 0.5
RIGID_REPEAT_COORDINATE_TOLERANCE = 0.002  # Angstrom; accommodates 0.001-A coordinate rounding.


class StructureEditError(ValueError):
    pass


class RepeatTransform(NamedTuple):
    """One fitted cycle; twist/rise are normalized per chain unit."""
    rotation: object
    translation: object
    axis: object
    twist: float
    rise: float
    rmsd: float
    diagnostics: dict


def automatic_water_axial_limit(detection):
    """Half the median physical layer spacing, plus 10% of that half-spacing."""
    from statistics import median
    spacings = [track.get('layer_spacing')
                for track in (detection or {}).get('protofilaments', [])]
    spacings = [float(value) for value in spacings
                if value is not None and math.isfinite(float(value)) and float(value) > 0]
    return 0.55 * median(spacings) if spacings else None


def _structure_dependencies():
    try:
        import gemmi
        import numpy as np
    except (ImportError, OSError) as exc:
        raise StructureEditError(
            "XCOP Modifier structure libraries could not be loaded. "
            "Run xcop_setup.py to install Gemmi and NumPy in the XCOP "
            "environment, then restart Modifier.\n\n"
            f"Details: {exc}"
        ) from exc
    return gemmi, np


def _cif_rows(block, category):
    data = block.get_mmcif_category(category)
    return [dict(zip(data, values)) for values in zip(*data.values())]


def _set_cif_rows(block, category, rows):
    block.find_mmcif_category(category).erase()
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        block.set_mmcif_category(category, {k: [r.get(k) for r in rows] for k in keys})


def _present(value):
    return value is not None and value is not False and value not in ("", ".", "?")


def _null_id(value):
    return str(value) if _present(value) else ""


def _cif_annotation_records(categories):
    if not categories:
        return []
    prefix = 'REMARK 999 AMYLOID_CIF '
    width = 80 - len(prefix)
    encoded = base64.b64encode(json.dumps(categories, ensure_ascii=False).encode('utf-8')).decode('ascii')
    return [prefix + encoded[i:i + width] for i in range(0, len(encoded), width)]


def structure_chain_id(index, max_length=None):
    # Bijective base 62 avoids truncating or reusing chain IDs.
    if not isinstance(index, int) or index < 0:
        raise StructureEditError("Chain index must be a nonnegative integer.")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    result = ""
    index += 1
    while index:
        index, digit = divmod(index - 1, len(alphabet))
        result = alphabet[digit] + result
    if max_length is not None and len(result) > max_length:
        raise StructureEditError("PDB chain namespace exhausted; export mmCIF.")
    return result


def _atom_key(row, model=True):
    fields = ("auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "label_comp_id",
              "label_atom_id", "label_alt_id")
    key = tuple(_null_id(row.get(k)) for k in fields)
    return ((_null_id(row.get("pdbx_PDB_model_num")) or "1",) + key) if model else key


def _residue_key(row):
    return tuple(_null_id(row.get(k)) for k in
                 ("auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "label_comp_id"))


def _endpoint_tests(row, prefix):
    # Unspecified CIF endpoint fields are wildcards.
    aliases = {
        "label_asym_id": (prefix + "label_asym_id",), "auth_asym_id": (prefix + "auth_asym_id",),
        "label_seq_id": (prefix + "label_seq_id",), "auth_seq_id": (prefix + "auth_seq_id",),
        "label_comp_id": (prefix + "label_comp_id",), "label_atom_id": (prefix + "label_atom_id",),
        "pdbx_PDB_ins_code": ("pdbx_" + prefix + "PDB_ins_code", prefix + "PDB_ins_code"),
        "label_alt_id": ("pdbx_" + prefix + "label_alt_id",),
    }
    return [(target, str(row[k])) for target, keys in aliases.items() for k in keys if _present(row.get(k))]


class _AtomIndex:

    def __init__(self, atoms):
        self.atoms = atoms
        self.indexes = {}

    def _candidates(self, tests):
        values = dict(tests)
        fields = next((fields for fields in (
            ('auth_asym_id', 'auth_seq_id', 'label_atom_id'),
            ('label_asym_id', 'label_seq_id', 'label_atom_id'),
            ('auth_asym_id', 'auth_seq_id'), ('label_asym_id', 'label_seq_id'),
            ('auth_asym_id',), ('label_asym_id',), ('label_atom_id',),
        ) if all(k in values for k in fields)), None)
        if fields is None:
            return self.atoms
        if fields not in self.indexes:
            index = defaultdict(list)
            for atom in self.atoms:
                index[tuple(str(atom.get(k)) for k in fields)].append(atom)
            self.indexes[fields] = index
        return self.indexes[fields].get(tuple(values[k] for k in fields), ())

    def matches(self, row, prefix):
        tests = _endpoint_tests(row, prefix)
        return [a for a in self._candidates(tests)
                if all(str(a.get(k)) == v for k, v in tests)] if tests else []

    def exists(self, row, prefix):
        tests = _endpoint_tests(row, prefix)
        return bool(tests) and any(all(str(a.get(k)) == v for k, v in tests)
                                   for a in self._candidates(tests))


class StructureBuffer:
    # Coordinates and edit provenance stay in memory until explicitly exported.

    def __init__(self, name, text='', report=None):
        self.name = os.path.basename(name)
        self.text = text
        self.report = deepcopy(report)
        self.component_styles = None
        self.active_alt_locs = None
        self.format = 'cif' if self.name.lower().endswith(('.cif', '.mmcif')) else 'pdb'

    def copy_from(self, other):
        self.text = other.text
        self.report = deepcopy(other.report)
        self.component_styles = deepcopy(other.component_styles)
        self.active_alt_locs = deepcopy(other.active_alt_locs)
        self.format = other.format


class StructureEditor:
    # PDB supports two-character IDs by default; strict mode uses one character.
    # Tolerate source annotation problems, keeping their contents in the report.

    # Categories whose meaning is independent of chain instances/coordinates.
    SAFE_CATEGORIES = set("""
        entry audit_conform audit_author citation citation_author citation_editor
        struct struct_keywords exptl exptl_crystal exptl_crystal_grow exptl_crystal_grow_comp
        cell symmetry space_group space_group_symop atom_type chem_comp chem_comp_atom
        chem_comp_bond chem_comp_angle chem_comp_tor chem_comp_chir chem_comp_plane
        chem_comp_plane_atom entity entity_poly entity_poly_seq entity_src_gen
        entity_src_nat pdbx_entity_src_syn pdbx_entity_nonpoly entity_name_com
        pdbx_entity_name_com pdbx_entity_keywords struct_ref struct_conn_type struct_conf_type
        pdbx_nmr_sample_details pdbx_nmr_exptl_sample pdbx_nmr_exptl_sample_conditions
        pdbx_nmr_exptl pdbx_nmr_refine pdbx_nmr_software pdbx_nmr_spectrometer
        em_experiment em_sample_preparation em_vitrification em_imaging em_detector
        em_image_recording em_3d_reconstruction em_helical_entity em_entity_assembly
        em_software software audit em_buffer em_sample_support em_ctf_correction
        em_entity_assembly_naturalsource em_image_processing em_specimen
        pdbx_contact_author pdbx_audit_support pdbx_nmr_details
        pdbx_initial_refinement_model em_image_scans em_buffer_component
        em_entity_assembly_molwt em_entity_assembly_recombinant em_imaging_optics
        em_particle_selection
    """.split())
    CHAIN_CATEGORIES = set("""
        struct_ref_seq struct_ref_seq_dif pdbx_struct_mod_residue
        pdbx_unobs_or_zero_occ_residues pdbx_unobs_or_zero_occ_atoms
        pdbx_poly_seq_scheme pdbx_nonpoly_scheme pdbx_branch_scheme
        struct_conf struct_mon_prot_cis struct_conn struct_site_gen
    """.split())
    SPECIAL_CATEGORIES = {'atom_site', 'atom_site_anisotrop', 'struct_asym', 'struct_sheet',
                          'struct_sheet_range', 'struct_sheet_order', 'pdbx_struct_sheet_hbond',
                          'struct_site', 'audit_syntax', 'amyloid_unrecognized_pdb'}
    INVALIDATED_PREFIXES = (
        "pdbx_validate_", "pdbx_vrpt_", "refine", "pdbx_refine", "struct_ncs",
        "pdbx_struct_assembly", "pdbx_struct_oper", "pdbx_audit_revision",
        "pdbx_database_", "database_", "pdbx_nmr_ensemble", "pdbx_nmr_representative",
        "struct_biol", "atom_sites", "pdbx_struct_special_symmetry", "pdbx_helical_symmetry", "em_3d_fitting",
        "pdbx_entry_details", "em_admin",
    )

    def __init__(self, path, *, metadata_policy="preserve", pdb_mode="extended"):
        gemmi, _ = _structure_dependencies()
        if metadata_policy not in ("error", "audit", "preserve") or pdb_mode not in ("extended", "strict"):
            raise StructureEditError("Invalid metadata policy or PDB mode.")
        buffer = path if isinstance(path, StructureBuffer) else None
        in_memory = buffer is not None or hasattr(path, 'read')
        if buffer is not None:
            self.path = 'memory:' + buffer.name
            text = buffer.text
            source_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        elif in_memory:
            self.path = str(getattr(path, 'name', '<memory>'))
            text = path.read()
            source_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        else:
            self.path = os.path.abspath(os.fspath(path))
            source_hash = self._hash(self.path)
            with open(self.path, encoding='utf-8-sig') as handle:
                text = handle.read()
        self.metadata_policy = metadata_policy
        self.pdb_mode = pdb_mode
        self.report = {"source": self.path, "source_sha256": source_hash,
                       "engine": "AmyloidModifier", "version": VERSION, "gemmi": gemmi.__version__,
                       "pdb_mode": pdb_mode, "warnings": [], "metadata": {}, "chain_copies": [],
                       "validation_scope": "syntax, identities, references and coordinate round-trip; not wwPDB deposition validation"}
        previous_path = self.path + '.audit.json'
        previous = deepcopy(buffer.report) if buffer is not None else None
        if previous is not None or (not in_memory and os.path.isfile(previous_path)):
            try:
                if previous is None:
                    with open(previous_path, encoding='utf-8') as handle:
                        previous = json.load(handle)
                if previous.get('output_sha256') == self.report['source_sha256']:
                    fields = ('engine', 'version', 'gemmi', 'operation', 'source', 'source_sha256',
                              'output', 'output_sha256', 'chain_copies', 'geometry', 'automatic_fit',
                              'overlap_check', 'terminal_completion', 'warnings',
                              'inferred_covalent_links', 'ensemble_selection')
                    self.report['history'] = previous.get('history', []) + [{k:previous[k] for k in fields if k in previous}]
                    archive = dict(previous.get('inherited_metadata_archive', {}))
                    archive.update({k:v for k,v in previous.get('metadata',{}).items() if 'source_content' in v})
                    if previous.get('atom_coordinate_metadata_archive'):
                        archive['atom_coordinate_metadata:' + previous.get('source_sha256', '')] = {
                            'source_content': previous['atom_coordinate_metadata_archive']}
                    self.report['inherited_metadata_archive'] = archive
                    self.report['source_issues'] = deepcopy(previous.get('source_issues', []))
                    self.report['warnings'].extend(issue['warning'] for issue in self.report['source_issues'])
                    if previous.get('source_pdb_metadata'):
                        self.report['source_pdb_metadata'] = previous['source_pdb_metadata']
                else:
                    self.report['warnings'].append('Previous audit hash differs from input; previous provenance was not inherited.')
            except (OSError, ValueError, TypeError, AttributeError):
                self.report['warnings'].append('Previous audit could not be read; no provenance was inferred from it.')
        first_token_line = next((line.lstrip() for line in text.splitlines()
                                 if line.strip() and not line.lstrip().startswith('#')), '')
        self.is_cif = first_token_line.lower().startswith('data_')
        self.pdb_lines = [] if self.is_cif else text.splitlines()
        self.report['source_record_counts'] = dict(Counter(l[:6].strip() for l in self.pdb_lines))
        known_pdb = set('HEADER OBSLTE TITLE SPLIT CAVEAT COMPND SOURCE KEYWDS EXPDTA NUMMDL MDLTYP AUTHOR REVDAT SPRSDE JRNL REMARK DBREF DBREF1 DBREF2 SEQADV SEQRES MODRES HET HETNAM HETSYN FORMUL HELIX SHEET SSBOND LINK CISPEP SITE CRYST1 ORIGX1 ORIGX2 ORIGX3 SCALE1 SCALE2 SCALE3 MTRIX1 MTRIX2 MTRIX3 MODEL ATOM HETATM ANISOU TER ENDMDL CONECT MASTER END'.split())
        unknown_pdb = set(self.report['source_record_counts']) - known_pdb - {''}
        self.unrecognized_pdb_records = [line for line in self.pdb_lines if line[:6].strip() in unknown_pdb]
        if unknown_pdb and self.metadata_policy == 'error':
            raise StructureEditError('PDB annotations without an edit handler: ' + ', '.join(sorted(unknown_pdb)) + '. Use audit policy to archive them explicitly.')
        if unknown_pdb:
            self._source_warning('unreviewed_pdb_records',
                'PDB annotations without an edit handler: ' + ', '.join(sorted(unknown_pdb)) +
                ('. These entries are preserved unchanged when saving; their references may describe the original structure.'
                 if self.metadata_policy == 'preserve' else
                 '. These annotations are archived in memory and omitted from the working copy; atom coordinates are retained.'),
                records=[{'line': i, 'text': line} for i, line in enumerate(self.pdb_lines, 1)
                         if line[:6].strip() in unknown_pdb])
        if self.is_cif:
            doc = gemmi.cif.read_string(text)
            if len(doc) != 1:
                raise StructureEditError("Select one mmCIF data block before editing; multiple blocks are ambiguous.")
            self.block = doc.sole_block()
            # Keep all categories, quoted values and multiline text in the CIF DOM.
            self.document = doc
            st = gemmi.make_structure_from_block(self.block)
        else:
            st = gemmi.read_pdb_string(text)
            self._check_pdb_atoms(st)
            st.setup_entities()
            st.assign_label_seq_id()
            self.document = st.make_mmcif_document(gemmi.MmcifOutputGroups(True, auth_all=True))
            self.block = self.document.sole_block()
            embedded = ''.join(line[len('REMARK 999 AMYLOID_CIF '):].strip()
                               for line in self.pdb_lines if line.startswith('REMARK 999 AMYLOID_CIF '))
            if embedded:
                try:
                    categories = json.loads(base64.b64decode(embedded, validate=True))
                    for category, rows in categories.items():
                        if not category.startswith('_') or not category.endswith('.') or not isinstance(rows, list):
                            raise ValueError('Invalid category envelope')
                        if self.block.find_mmcif_category(category):
                            raise ValueError('Embedded metadata conflicts with parsed coordinates or metadata')
                        _set_cif_rows(self.block, category, rows)
                except (ValueError, TypeError, AttributeError) as exc:
                    raise StructureEditError('Cannot restore embedded mmCIF annotations: ' + str(exc)) from exc
        if self.metadata_policy == 'preserve':
            if self.unrecognized_pdb_records:
                _set_cif_rows(self.block, '_amyloid_unrecognized_pdb.',
                              [{'id': str(i), 'text': line} for i, line in enumerate(self.unrecognized_pdb_records, 1)])
            else:
                self.unrecognized_pdb_records = [row['text'] for row in _cif_rows(self.block, '_amyloid_unrecognized_pdb.')]
        self.atoms = _cif_rows(self.block, "_atom_site.")
        if not self.atoms:
            raise StructureEditError("No atom_site coordinates found.")
        required = {"id", "label_asym_id", "label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"}
        if not required <= self.atoms[0].keys():
            raise StructureEditError("Missing required atom_site columns: " + ", ".join(sorted(required - self.atoms[0].keys())))
        for row in self.atoms:
            for auth, label in (("auth_asym_id", "label_asym_id"), ("auth_seq_id", "label_seq_id"),
                                ("auth_atom_id", "label_atom_id"), ("auth_comp_id", "label_comp_id")):
                if not _present(row.get(auth)):
                    row[auth] = row.get(label)
            row.setdefault("pdbx_PDB_model_num", "1")
        self.models = list(dict.fromkeys(str(r["pdbx_PDB_model_num"]) for r in self.atoms))
        self.chains = list(dict.fromkeys(_null_id(r["auth_asym_id"]) for r in self.atoms))
        self._validate_atoms(self.atoms)
        self.poly_labels = {r["id"] for r in _cif_rows(self.block, "_struct_asym.")
                            if r.get("entity_id") in {e["entity_id"] for e in _cif_rows(self.block, "_entity_poly.")}}
        if not self.poly_labels:
            self.poly_labels = {r["label_asym_id"] for r in self.atoms if _present(r.get("label_seq_id"))}
        polymer_types = {e['entity_id']: e.get('type', '') for e in _cif_rows(self.block, '_entity_poly.')}
        label_types = {r['id']: polymer_types.get(r.get('entity_id'), '')
                       for r in _cif_rows(self.block, '_struct_asym.')}
        ca_labels = {r['label_asym_id'] for r in self.atoms
                     if r['label_atom_id'] == 'CA' and r.get('type_symbol', '').upper() == 'C'}
        # Polymer metadata also describes RNA/DNA ligands. Keep that metadata,
        # but only peptide chains belong to the C-alpha protein scaffold.
        self.protein_labels = {label for label in self.poly_labels
                               if label_types.get(label, '').startswith('polypeptide') or
                               (not label_types.get(label) and label in ca_labels)}
        self.pdb_bonds = self._read_conect(st) if not self.is_cif else []
        self._check_source_connections()
        self.groups = []
        self.output_atoms = []
        self.output_block = None

    @classmethod
    def from_string(cls, text, *, source_name='<memory>', **options):
        with StringIO(text) as source:
            source.name = source_name
            return cls(source, **options)

    @staticmethod
    def _hash(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def _source_warning(self, kind, message, **details):
        prefix = ('Source annotation notice: ' if kind in ('unreviewed_pdb_records', 'unreviewed_cif_category')
                  else 'Non-standard source: ')
        warning = prefix + message
        issue = dict(kind=kind, source=self.path, source_sha256=self.report['source_sha256'],
                     warning=warning, **details)
        issues = self.report.setdefault('source_issues', [])
        if not any(old['kind'] == kind and old['warning'] == warning for old in issues):
            issues.append(issue)
        if warning not in self.report['warnings']:
            self.report['warnings'].append(warning)

    def unrecognized_cif_categories(self):
        return {category: _cif_rows(self.block, category)
                for category in self.block.get_mmcif_category_names()
                if category.strip('_.') not in self.SAFE_CATEGORIES | self.CHAIN_CATEGORIES | self.SPECIAL_CATEGORIES
                and not category.strip('_.').startswith(self.INVALIDATED_PREFIXES)}

    def _check_source_connections(self):
        rows = _cif_rows(self.block, '_struct_conn.')
        index = _AtomIndex(self.atoms)
        kept, missing = [], []
        for row in rows:
            (kept if all(index.exists(row, prefix) for prefix in ('ptnr1_', 'ptnr2_'))
             else missing).append(row)
        if missing:
            if self.metadata_policy == 'error':
                raise StructureEditError('Dangling struct_conn atom endpoint in source.')
            self._source_warning('dangling_struct_conn',
                f'{len(missing)} connection annotation(s) refer to missing atoms. '
                'Ignored these annotations in the working copy; atom coordinates and valid connections are retained.',
                rows=missing)
            _set_cif_rows(self.block, '_struct_conn.', kept)

    def _check_pdb_atoms(self, st):
        seen, anisou, model = set(), [], "1"
        serial_lines, identities = defaultdict(list), set()
        atom_count = 0
        for line_number, line in enumerate(self.pdb_lines, 1):
            rec = line[:6].strip()
            if rec == "MODEL":
                model = line[10:14].strip()
            elif rec in ("ATOM", "HETATM", "ANISOU"):
                if len(line) < 54:
                    raise StructureEditError("Truncated PDB coordinate/ANISOU record.")
                if rec == "ANISOU":
                    anisou.append((model, line[6:11], line[12:27]))
                    continue
                key = (model, line[6:11])
                if key in seen and self.metadata_policy == 'error':
                    raise StructureEditError("Duplicate atom serial within PDB MODEL: " + str(key))
                seen.add(key)
                serial_lines[key].append(line_number)
                identities.add((model, line[6:11], line[12:27]))
                atom_count += 1
                for start in (30, 38, 46):
                    if not math.isfinite(float(line[start:start + 8])):
                        raise StructureEditError("Nonfinite PDB coordinates.")
        if atom_count != sum(m.count_atom_sites() for m in st):
            raise StructureEditError("PDB parser did not retain every atom; resolve input ambiguity first.")
        if any(identity not in identities for identity in anisou):
            raise StructureEditError("ANISOU does not match its atom serial and identity.")
        duplicates = [{'model': m, 'serial': s.strip(), 'lines': lines}
                      for (m, s), lines in serial_lines.items() if len(lines) > 1]
        if duplicates:
            self._source_warning('duplicate_pdb_serials',
                f"{sum(len(item['lines']) for item in duplicates)} atoms share repeated PDB serial numbers. "
                'Atom identities and coordinates are retained; serials are regenerated in the working copy. '
                'Ambiguous CONECT references are ignored.', serials=duplicates)

    def _read_conect(self, st):
        # Read fixed-width fields, including adjacent five-digit serials.
        pairs = Counter()
        pair_records = defaultdict(list)
        for line_number, line in enumerate(self.pdb_lines, 1):
            if not line.startswith("CONECT"):
                continue
            try:
                nums = [int(line[i:i + 5]) for i in range(6, min(len(line), 31), 5) if line[i:i + 5].strip()]
            except ValueError as exc:
                raise StructureEditError("CONECT with nondecimal serials is not supported; convert to mmCIF first.") from exc
            if len(nums) > 1:
                pairs.update((nums[0], target) for target in nums[1:])
                for target in set(nums[1:]):
                    pair_records[(nums[0], target)].append({'line': line_number, 'text': line})
        serial_to_id = {}
        for m in st:
            for c in m:
                for res in c:
                    for atom in res:
                        serial_to_id.setdefault(atom.serial, []).append(
                            (str(m.num), c.name, str(res.seqid.num), _null_id(res.seqid.icode.strip()),
                             res.name, atom.name, atom.altloc.replace("\x00", "")))
        ids = {_atom_key(r): r["id"] for r in self.atoms}
        bonds, missing, ambiguous = {}, [], []
        for (a, b), order in pairs.items():
            common_models = ({key[0] for key in serial_to_id.get(a, ())} &
                             {key[0] for key in serial_to_id.get(b, ())})
            if not common_models:
                if self.metadata_policy == 'error':
                    raise StructureEditError(f"Dangling CONECT reference {a} -> {b} in source.")
                missing.append({'from': a, 'to': b, 'order': order, 'records': pair_records[(a, b)]})
                continue
            ambiguous_models = {m for m in common_models
                                if sum(key[0] == m for key in serial_to_id[a]) != 1
                                or sum(key[0] == m for key in serial_to_id[b]) != 1}
            if ambiguous_models:
                ambiguous.append({'from': a, 'to': b, 'models': sorted(ambiguous_models),
                                  'order': order, 'records': pair_records[(a, b)]})
            for ka in serial_to_id[a]:
                if ka[0] in ambiguous_models:
                    continue
                for kb in serial_to_id[b]:
                    if ka[0] == kb[0]:
                        key = tuple(sorted((ids[ka], ids[kb]), key=int))
                        bonds[key] = max(bonds.get(key, 0), order)
        if missing:
            examples = ', '.join(f"{ref['from']} -> {ref['to']}" for ref in missing[:5])
            if len(missing) > 5:
                examples += ', ...'
            self._source_warning('dangling_conect',
                f'{len(missing)} dangling CONECT reference(s) ({examples}). '
                'Ignored these references in the working copy because the atoms do not exist in the same model; '
                'atom coordinates and valid bonds are retained.', references=missing)
        if ambiguous:
            self._source_warning('ambiguous_conect',
                f'{len(ambiguous)} CONECT reference(s) use repeated atom serials. '
                'Ignored these references in affected models instead of guessing bond endpoints.',
                references=ambiguous)
        return [(a, b, n) for (a, b), n in bonds.items()]

    @staticmethod
    def _validate_atoms(atoms):
        identities, serials = set(), set()
        label_entities, label_auth = {}, {}
        for row in atoms:
            key = _atom_key(row)
            if key in identities:
                raise StructureEditError("Duplicate atom identity (including model, insertion code and altloc): " + str(key))
            identities.add(key)
            if row["id"] in serials:
                raise StructureEditError("atom_site.id must be unique across the mmCIF block.")
            serials.add(row["id"])
            for field in ("Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv"):
                if _present(row.get(field)) and not math.isfinite(float(row[field])):
                    raise StructureEditError("Nonfinite atom value in " + field)
            if _present(row.get("occupancy")) and not 0 <= float(row["occupancy"]) <= 1.00001:
                raise StructureEditError("Occupancy outside [0, 1].")
            label = row["label_asym_id"]
            if not _present(label):
                raise StructureEditError("Every atom needs a label_asym_id.")
            auth = row["auth_asym_id"]
            if label in label_auth and label_auth[label] != auth:
                raise StructureEditError("One label_asym_id refers to multiple author chains.")
            label_auth[label] = auth
            entity = row.get("label_entity_id")
            if label in label_entities and label_entities[label] != entity:
                raise StructureEditError("One label_asym_id refers to multiple entities.")
            label_entities[label] = entity

    def ca_chains(self):
        # Choose one CA conformer per residue by occupancy, using the first model.
        _, np = _structure_dependencies()
        candidates = defaultdict(dict)
        for row in self.atoms:
            if str(row["pdbx_PDB_model_num"]) != self.models[0]:
                continue
            if row["label_asym_id"] not in self.protein_labels or row["label_atom_id"] != "CA" or row.get("type_symbol", "C").upper() != "C":
                continue
            key = _residue_key(row)[1:]
            chain = _null_id(row["auth_asym_id"])
            old = candidates[chain].get(key)
            rank = (float(row.get("occupancy") or 0), _null_id(row.get("label_alt_id")) in ("", "A"))
            if old is None or rank > old[0]:
                candidates[chain][key] = (rank, row)
        return {c: [{"id": (k[0] + k[1]), "key": k,
                      "sequence_key": ((str(r['label_entity_id']), str(r['label_seq_id']), r['label_comp_id'])
                                       if _present(r.get('label_entity_id')) and _present(r.get('label_seq_id')) else None),
                      "coord": np.array([float(r[f]) for f in ("Cartn_x", "Cartn_y", "Cartn_z")])}
                    for k, (_, r) in residues.items()] for c, residues in candidates.items()}

    def preview(self):
        waters, others = [], []
        for r in self.atoms:
            if str(r["pdbx_PDB_model_num"]) != self.models[0] or r["label_asym_id"] in self.protein_labels:
                continue
            xyz = tuple(float(r[f]) for f in ("Cartn_x", "Cartn_y", "Cartn_z"))
            if r["label_comp_id"] in ("HOH", "WAT", "DOD"):
                if r["type_symbol"].upper() == "O":
                    waters.append(xyz)
            else:
                others.append(xyz)
        return self.ca_chains(), waters, others

    def companion_groups(self):
        # Keep a nonprotein polymer, and any explicitly covalently connected
        # nonprotein residues, together as a complete companion molecule.
        groups = defaultdict(list)
        for row in self.atoms:
            label = row['label_asym_id']
            if str(row['pdbx_PDB_model_num']) != self.models[0] or label in self.protein_labels:
                continue
            key = ('polymer', label) if label in self.poly_labels else ('residue', _residue_key(row))
            groups[key].append(row)
        by_atom = {r['id']: key for key, rows in groups.items() for r in rows}
        parent = {key: key for key in groups}
        def root(key):
            while parent[key] != key:
                key = parent[key]
            return key
        def join(a, b):
            if a in by_atom and b in by_atom:
                ra, rb = root(by_atom[a]), root(by_atom[b])
                if ra != rb:
                    parent[rb] = ra
        for a, b, _ in self.pdb_bonds:
            join(a, b)
        connections = [r for r in _cif_rows(self.block, '_struct_conn.')
                       if r.get('conn_type_id') in ('covale', 'disulf', 'modres')]
        if connections:
            index = _AtomIndex([r for rows in groups.values() for r in rows])
            for row in connections:
                for a in index.matches(row, 'ptnr1_'):
                    for b in index.matches(row, 'ptnr2_'):
                        join(a['id'], b['id'])
        merged = defaultdict(list)
        for key, rows in groups.items():
            merged[root(key)].extend(rows)
        return list(merged.values())

    def associated_residues(self, core_chains, axial_limit=4.0, axis=None):
        # Assign complete companion molecules once, to the nearest protein chain.
        _, np = _structure_dependencies()
        axis = np.asarray(axis if axis is not None else (0, 0, 1), dtype=float)
        axis /= np.linalg.norm(axis)
        ca = self.ca_chains()
        centers = {c: np.mean([r["coord"] for r in ca[c]], axis=0) for c in core_chains if c in ca}
        owners = {}
        for rows in self.companion_groups():
            pos = np.mean([[float(r[f]) for f in ("Cartn_x", "Cartn_y", "Cartn_z")] for r in rows], axis=0)
            choices = [(float(np.linalg.norm(pos - cen)), c) for c, cen in centers.items()
                       if abs(float(np.dot(pos - cen, axis))) <= axial_limit]
            if choices:
                owners.update((_residue_key(r), min(choices)[1]) for r in rows)
        return owners

    def covalent_residue_owners(self):
        # Keep covalently attached nonpolymers with their protein chain.
        first = [r for r in self.atoms if str(r['pdbx_PDB_model_num']) == self.models[0]]
        index = _AtomIndex(first)
        by_id = {r['id']: r for r in first}
        pairs = [(by_id[a], by_id[b]) for a, b, _ in self.pdb_bonds if a in by_id and b in by_id]
        for row in _cif_rows(self.block, '_struct_conn.'):
            if row.get('conn_type_id') in ('covale', 'disulf', 'modres'):
                pairs.extend((a, b) for a in index.matches(row, 'ptnr1_') for b in index.matches(row, 'ptnr2_'))
        links, owners = defaultdict(set), defaultdict(set)
        for a, b in pairs:
            ka, kb = _residue_key(a), _residue_key(b)
            pa, pb = a['label_asym_id'] in self.protein_labels, b['label_asym_id'] in self.protein_labels
            if not pa and not pb:
                links[ka].add(kb)
                links[kb].add(ka)
            elif pa != pb:
                owners[kb if pa else ka].add(a['auth_asym_id'] if pa else b['auth_asym_id'])
        for rows in self.companion_groups():
            keys = list(dict.fromkeys(_residue_key(r) for r in rows))
            for a, b in zip(keys, keys[1:]):
                links[a].add(b)
                links[b].add(a)
        pending = list(owners)
        while pending:
            key = pending.pop()
            for other in links[key]:
                added = owners[key] - owners[other]
                if added:
                    owners[other].update(added)
                    pending.append(other)
        if any(len(chains) > 1 for chains in owners.values()):
            raise StructureEditError('A nonpolymer is covalently attached to multiple protein chains; expand the connected assembly explicitly.')
        return {key: next(iter(chains)) for key, chains in owners.items() if chains}

    def associated_residue_repeats(self, sandwiches, axis, rise, axial_limit=4.0):
        # Identify ligand sites and their axial occupancy periods.
        _, np = _structure_dependencies()
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        rise = abs(float(rise))
        locations = {c: (pf, i) for pf, stack in enumerate(sandwiches) for i, c in enumerate(stack)}
        owners = self.associated_residues(set(locations), axial_limit, axis)
        sites = []
        for rows in self.companion_groups():
            keys = tuple(dict.fromkeys(_residue_key(r) for r in rows))
            key = keys[0]
            if key not in owners:
                continue
            xyz = np.array([[float(r[f]) for f in ('Cartn_x', 'Cartn_y', 'Cartn_z')] for r in rows])
            center = xyz.mean(0)
            pf, layer = locations[owners[key]]
            entry = dict(key=key, keys=keys, layer=layer, center=center,
                         extent=float(np.ptp(xyz @ axis)), owner=owners[key],
                         xyz=xyz, polymer=rows[0]['label_asym_id'] in self.poly_labels)
            # Atom order must not affect the orientation comparison between copies.
            entry['xyz'] = xyz[sorted(range(len(rows)), key=lambda i:
                (keys.index(_residue_key(rows[i])), rows[i]['label_atom_id'],
                 _null_id(rows[i].get('label_alt_id'))))]
            identity = ('+'.join(k[-1] for k in keys),
                        frozenset((keys.index(_residue_key(r)), r['label_atom_id']) for r in rows))
            if len(keys) > 1:
                # An axial oligonucleotide occupies one repeat per nucleotide.
                # Atomic extent includes terminal base/ribose overhangs, so use
                # successive residue centers to estimate its whole-chain period.
                projections = np.array([xyz[[_residue_key(r) == k for r in rows]].mean(0) @ axis for k in keys])
                spacing = np.abs(np.diff(projections))
                if np.all((spacing > .5 * rise) & (spacing < 1.5 * rise)):
                    entry['polymer_period'] = max(1, int(round((np.ptp(projections) + np.median(spacing)) / rise)))
            # Nearest protein centroids can switch protofilaments at a terminal
            # overhang. A molecular track is identified by its own transverse
            # position and axial separation, not that clipped owner assignment.
            site = next((s for s in sites if s['identity'] == identity and
                         (s['pf'] == pf or (len(xyz) >= 6 and not entry['polymer'] and
                                           not any(r['polymer'] for r in s['residues']))) and
                         all((abs(float((center - r['center']) @ axis)) > .5 * rise
                              if len(xyz) >= 6 else layer != r['layer'])
                             for r in s['residues']) and
                         any(np.linalg.norm((center - r['center']) - np.dot(center - r['center'], axis) * axis) <= 4.0
                             for r in s['residues'])), None)
            if site is None:
                site = dict(identity=identity, pf=pf, residues=[])
                sites.append(site)
            site['residues'].append(entry)
        for site in sites:
            members = sorted(site['residues'], key=lambda r: r['layer'])
            gaps = [b['layer'] - a['layer'] for a, b in zip(members, members[1:])]
            extent = float(np.median([r['extent'] for r in members]))
            inferred = members[0].get('polymer_period', 1 if extent < rise else int(math.ceil(extent / rise)) + 1)
            if gaps and (len(members[0]['xyz']) >= 6 or all('polymer_period' in r for r in members)):
                # Terminal companions can extend beyond the protein bounds;
                # measure separation rather than a clipped nearest-owner index.
                positions = sorted(float(r['center'] @ axis) for r in members)
                gaps = [max(1, int(round((b - a) / rise))) for a, b in zip(positions, positions[1:])]
            site['period'] = max(1, min(gaps)) if gaps else inferred
            # Repeated small molecules can overhang a terminal protein layer.
            # Recover their positions from molecular separation, not the clipped
            # nearest-protein index. Preserve the observed coverage at each end.
            ordered = sorted(members, key=lambda r: float(r['center'] @ axis))
            positions = [float(r['center'] @ axis) for r in ordered]
            site['observed_track'] = (len(ordered) >= 2 and
                len(ordered[0]['xyz']) >= 6 and
                not any(r['polymer'] for r in ordered) and site['period'] == 1 and
                all(abs((b - a) / rise - 1) < .25 for a, b in zip(positions, positions[1:])))
            if site['observed_track']:
                first_layer = ordered[0]['layer']
                for offset, residue in enumerate(ordered):
                    residue['layer'] = first_layer + offset
            # Endpoint coverage can only be extrapolated when the observed
            # ligand track covers essentially the entire protein stack. A short
            # internal patch must not be treated as if it reached both ends.
            site['terminal_track'] = (site['observed_track'] and
                len(ordered) >= len(sandwiches[site['pf']]) - 1)
        phases = defaultdict(Counter)
        for site in sites:
            period = site['period']
            if period > 1:
                for residue in site['residues']:
                    phases[site['identity'], period][residue['layer'] % period] += 1
        for site in sites:
            period = site['period']
            if period > 1:
                phase = phases[site['identity'], period].most_common(1)[0][0]
                for residue in site['residues']:
                    layer = phase + period * round((residue['layer'] - phase) / period)
                    residue['layer'] = min(len(sandwiches[site['pf']]) - 1, max(0, layer))
        return sites

    def associated_residues_for_layers(self, sandwiches, keep_indices, axis, axial_limit=4.0,
                                       *, keep_chains=None):
        # Keep complete ligand repeats when trimming protein layers.
        _, np = _structure_dependencies()
        if keep_chains is None:
            indices = set(keep_indices)
            kept_by_pf = [{i for i in indices if 0 <= i < len(s)} for s in sandwiches]
        else:
            keep_chains = set(keep_chains)
            kept_by_pf = [{i for i, c in enumerate(s) if c in keep_chains} for s in sandwiches]
        core = {c for s in sandwiches for c in s}
        kept_chains = {s[i] for s, kept in zip(sandwiches, kept_by_pf) for i in kept}
        owners = self.associated_residues(core, axial_limit, axis)
        attached = self.covalent_residue_owners()
        owners.update(attached)
        result = {r for r, c in owners.items() if c in kept_chains}
        if not kept_chains or axis is None or not any(len(s) >= 2 for s in sandwiches):
            return result
        ca = self.ca_chains()
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        # Measure within each stack: unequal ends must not mix unrelated indices
        # or shift a layer centroid when a shorter protofilament runs out.
        spacings = []
        for stack in sandwiches:
            centers = np.array([np.mean([r['coord'] for r in ca[c]], axis=0) for c in stack])
            spacings.extend(np.abs(np.diff(centers @ axis)))
        rise = float(np.median(spacings))
        if rise < 1e-8:
            return result
        sites = self.associated_residue_repeats(sandwiches, axis, rise, axial_limit)
        for site in sites:
            kept = kept_by_pf[site['pf']]
            depth = len(sandwiches[site['pf']])
            if not kept:
                result.difference_update(k for r in site['residues'] for k in r['keys'])
                continue
            midpoint = float(np.mean(sorted(kept)))
            if site.get('terminal_track') and kept == set(range(min(kept), max(kept) + 1)):
                members = sorted(site['residues'], key=lambda r: r['layer'])
                result.difference_update(k for r in members for k in r['keys'])
                end = max(0, len(members) - (depth - 1 - max(kept)))
                result.update(k for r in members[min(kept):end] for k in r['keys'])
                continue
            if site['period'] == 1:
                continue
            members = site['residues']
            result.difference_update(k for r in members for k in r['keys'])
            removed_repeats = depth // site['period'] - len(kept) // site['period']
            count = max(1, len(members) - removed_repeats)
            eligible = [r for r in members if min(abs(r['layer'] - i) for i in kept) <= site['period'] / 2]
            eligible.sort(key=lambda r: (abs(r['layer'] - midpoint), r['layer']))
            result.update(k for r in eligible[:count] for k in r['keys'])
        result.difference_update(attached)
        result.update(r for r, c in attached.items() if c in kept_chains)
        return result

    def trim(self, keep_chains, *, rename=None, keep_residues=(), polymer_only=False):
        keep = set(keep_chains)
        unknown = keep - set(self.chains)
        if unknown or not keep:
            raise StructureEditError("Empty selection or unknown chains: " + ", ".join(sorted(unknown)))
        residues = set(keep_residues)
        rows = [r for r in self.atoms if (_null_id(r["auth_asym_id"]) in keep and
                (not polymer_only or r['label_asym_id'] in self.protein_labels)) or _residue_key(r) in residues]
        mapping = {c: (rename or {}).get(c, c) for c in self.chains if any(_null_id(r["auth_asym_id"]) == c for r in rows)}
        self.report["operation"] = "trim/rename"
        return self.apply_copies([{"chains": mapping, "atom_ids": {r["id"] for r in rows}}])

    def rename(self, mapping):
        if set(mapping) - set(self.chains):
            raise StructureEditError("Rename map contains unknown author chain IDs.")
        self.report["operation"] = "rename"
        return self.apply_copies([{"chains": {c: mapping.get(c, c) for c in self.chains}}])

    def select_model(self, model_num, output_model="1"):
        # Select a source ensemble member before editing or merging live coordinates.
        model_num, output_model = str(model_num), str(output_model)
        if self.output_block is not None or model_num not in self.models:
            raise StructureEditError('Select an existing source model before editing.')
        original_models = list(self.models)
        self.atoms = [r for r in self.atoms if str(r['pdbx_PDB_model_num']) == model_num]
        atom_ids = {r['id'] for r in self.atoms}
        self.pdb_bonds = [(a, b, order) for a, b, order in self.pdb_bonds if a in atom_ids and b in atom_ids]
        unrecognized = self.unrecognized_cif_categories() if self.metadata_policy == 'preserve' else {}
        for category in self.block.get_mmcif_category_names():
            if category in unrecognized:
                continue
            rows = _cif_rows(self.block, category)
            if not rows:
                continue
            fields = [k for k in rows[0] if k.lower() in ('pdbx_pdb_model_num', 'pdb_model_num')]
            if not fields and category != '_atom_site_anisotrop.':
                continue
            kept = []
            for row in rows:
                if category == '_atom_site_anisotrop.' and row['id'] not in atom_ids:
                    continue
                if any(_present(row.get(k)) and str(row[k]) != model_num for k in fields):
                    continue
                kept.append(dict(row, **{k: output_model for k in fields if _present(row.get(k))}))
            _set_cif_rows(self.block, category, kept)
        for row in self.atoms:
            row['pdbx_PDB_model_num'] = output_model
        self.models = [output_model]
        self.chains = list(dict.fromkeys(_null_id(r['auth_asym_id']) for r in self.atoms))
        self.report['ensemble_selection'] = {'source_models': original_models,
            'selected_source_model': model_num, 'working_model': output_model}
        self.report['warnings'].append(f'Working copy contains selected ensemble member {model_num} of {len(original_models)}; the other source members were not combined into layers.')
        return self

    def apply_copies(self, copies):
        gemmi, np = _structure_dependencies()
        self.report.setdefault("operation", "explicit rigid copies")
        self.groups, self.output_atoms = [], []
        used_auth = set()
        next_label = 0
        for number, spec in enumerate(copies):
            mapping = dict(spec["chains"])
            if set(mapping) - set(self.chains):
                raise StructureEditError("Copy refers to an unknown source chain.")
            if len(set(mapping.values())) != len(mapping) or used_auth.intersection(mapping.values()):
                raise StructureEditError("Output author chain IDs must be unique; merging chains is not a rename.")
            if any(not isinstance(c, str) or not c or not c.isascii() or not c.isalnum() for c in mapping.values()):
                raise StructureEditError("Output chain IDs must be nonempty ASCII alphanumeric strings.")
            used_auth.update(mapping.values())
            rotation = np.asarray(spec.get("matrix", np.eye(3)), dtype=float)
            translation = np.asarray(spec.get("vector", np.zeros(3)), dtype=float)
            if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                raise StructureEditError("Rigid transform must contain finite 3x3 R and length-3 t.")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
                raise StructureEditError("Transform must be a proper rotation (det R = +1), without scale/reflection.")
            selection = spec.get("atom_ids")
            rows = [r for r in self.atoms if _null_id(r["auth_asym_id"]) in mapping and (selection is None or r["id"] in selection)]
            if not rows:
                raise StructureEditError("A requested copy contains no atoms.")
            labels, ids, output_rows = {}, {}, []
            rotated_coordinates = not np.allclose(rotation, np.eye(3), atol=1e-12, rtol=0)
            identity_coordinates = np.array_equal(rotation, np.eye(3)) and np.array_equal(translation, np.zeros(3))
            for row in rows:
                old = row["label_asym_id"]
                if old not in labels:
                    # Ligands sharing a polymer author ID still need independent label/entity IDs.
                    labels[old] = structure_chain_id(next_label)
                    next_label += 1
                new = dict(row)
                new["id"] = str(len(self.output_atoms) + 1)
                ids[row["id"]] = new["id"]
                new["auth_asym_id"] = mapping[_null_id(row["auth_asym_id"])]
                new["label_asym_id"] = labels[old]
                if not identity_coordinates:
                    xyz = rotation @ np.array([float(row[f]) for f in ("Cartn_x", "Cartn_y", "Cartn_z")]) + translation
                    for field, value in zip(("Cartn_x", "Cartn_y", "Cartn_z"), xyz):
                        new[field] = f"{value:.6f}"
                # Drop frame-dependent uncertainties that cannot be rotated without covariances.
                uncertainty_fields = {"Cartn_x_esd", "Cartn_y_esd", "Cartn_z_esd"}
                invalidated = {field: value for field, value in row.items()
                               if _present(value) and (field.startswith('fract_') or
                                   (rotated_coordinates and field in uncertainty_fields))}
                if invalidated:
                    for field in invalidated:
                        new[field] = None
                    self.report.setdefault('atom_coordinate_metadata_archive', {}).setdefault(row['id'], {}).update(invalidated)
                    self.report['warnings'].append('Fractional coordinate metadata and rotated Cartesian ESDs were invalidated where present; source values are archived by atom ID.')
                self.output_atoms.append(new)
                output_rows.append(new)
            group = {"chains": mapping, "labels": labels, "ids": ids, "rotation": rotation,
                     "translation": translation, "rows": rows, "index": number,
                     "output_rows": output_rows, "atom_index": _AtomIndex(output_rows)}
            self.groups.append(group)
            self.report["chain_copies"].append({"copy": number, "author_chains": mapping,
                "label_chains": labels, "matrix": rotation.tolist(), "vector": translation.tolist()})
        if not self.output_atoms:
            raise StructureEditError("Refusing to write an empty structure.")
        self._validate_atoms(self.output_atoms)
        self.output_index = _AtomIndex(self.output_atoms)
        self.output_document = gemmi.cif.Document()
        self.output_block = self.output_document.add_new_block("AmyloidModifier")
        self._metadata()
        _set_cif_rows(self.output_block, "_atom_site.", self.output_atoms)
        self._anisotrop()
        self._conect_to_connections()
        self._collect_declared_bonds()
        self._validate_references()
        return self

    def _collect_declared_bonds(self):
        by_model = defaultdict(list)
        by_residue = defaultdict(list)
        for atom in self.output_atoms:
            by_model[str(atom['pdbx_PDB_model_num'])].append(atom)
            if atom.get('group_PDB') == 'HETATM':
                by_residue[(str(atom['pdbx_PDB_model_num']), _residue_key(atom))].append(atom)
        model_indexes = [_AtomIndex(atoms) for atoms in by_model.values()]
        def order(row):
            return {'sing':1,'doub':2,'trip':3}.get(str(row.get('pdbx_value_order', row.get('value_order','sing'))).lower(),1)
        for connection in _cif_rows(self.output_block, '_struct_conn.'):
            if connection.get('conn_type_id') not in ('covale','disulf','modres','metalc'):
                continue
            for index in model_indexes:
                partners = [index.matches(connection, prefix) for prefix in ('ptnr1_','ptnr2_')]
                for a in partners[0]:
                    for b in partners[1]:
                        alt_a, alt_b = _null_id(a.get('label_alt_id')), _null_id(b.get('label_alt_id'))
                        if alt_a and alt_b and alt_a != alt_b:
                            continue
                        self.output_bonds.append((a['id'],b['id'],order(connection)))
        bonds = defaultdict(list)
        for row in _cif_rows(self.output_block, '_chem_comp_bond.'):
            bonds[row['comp_id']].append(row)
        for (_, residue), atoms in by_residue.items():
            for row in bonds.get(residue[-1], []):
                a_atoms = [a for a in atoms if a['label_atom_id'] == row['atom_id_1']]
                b_atoms = [a for a in atoms if a['label_atom_id'] == row['atom_id_2']]
                for a in a_atoms:
                    for b in b_atoms:
                        aa, ab = _null_id(a.get('label_alt_id')), _null_id(b.get('label_alt_id'))
                        if aa and ab and aa != ab:
                            continue
                        self.output_bonds.append((a['id'],b['id'],order(row)))
        unique = {}
        for a,b,n in self.output_bonds:
            key = tuple(sorted((a,b),key=int))
            unique[key] = max(n,unique.get(key,0))
        self.output_bonds = [(a,b,n) for (a,b),n in unique.items()]

    def _audit(self, category, action, reason="", content=None):
        item = {"action": action, "reason": reason}
        if content is not None:
            item["source_content"] = content
        self.report["metadata"][category] = item

    def _remap_row(self, row, group, *, extra_label=(), extra_auth=()):
        out = dict(row)
        for key, value in row.items():
            lower = key.lower()
            label = ("asym_id" in lower and "auth" not in lower) or key in extra_label
            auth = ("auth_asym_id" in lower) or key in extra_auth or lower in ("pdbx_strand_id", "pdb_strand_id", "pdbx_pdb_strand_id")
            if not (label or auth) or not _present(value):
                continue
            mapping = group["chains"] if auth else group["labels"]
            # Chain-list fields have list semantics; partners/endpoints do not.
            if lower in ("pdbx_strand_id", "pdb_strand_id", "pdbx_pdb_strand_id"):
                values = [mapping[c.strip()] for c in str(value).split(",") if c.strip() in mapping]
                if not values:
                    return None
                out[key] = ",".join(dict.fromkeys(values))
            elif value not in mapping:
                return None
            else:
                out[key] = mapping[value]
        return out

    @staticmethod
    def _endpoint_exists(row, atoms, prefix):
        if isinstance(atoms, _AtomIndex):
            return atoms.exists(row, prefix)
        tests = _endpoint_tests(row, prefix)
        return bool(tests) and any(all(str(a.get(k)) == v for k, v in tests) for a in atoms)

    def _metadata(self):
        out = self.output_block
        special = {"atom_site", "atom_site_anisotrop", "struct_asym", "struct_sheet", "struct_sheet_range",
                   "struct_sheet_order", "pdbx_struct_sheet_hbond", "struct_site"}
        live_entities = {r.get("label_entity_id") for r in self.output_atoms if _present(r.get("label_entity_id"))}
        align_ids, site_ids = {}, {}
        for group in self.groups:
            for row in _cif_rows(self.block, '_struct_ref_seq.'):
                if self._remap_row(row, group) is not None:
                    align_ids[group['index'], row['align_id']] = str(len(align_ids) + 1)
            for row in _cif_rows(self.block, '_struct_site_gen.'):
                if self._remap_row(row,group) is not None:
                    key = (group['index'],row['site_id'])
                    site_ids.setdefault(key, str(len(site_ids)+1))
        for category in self.block.get_mmcif_category_names():
            name = category.strip("_.")
            rows = _cif_rows(self.block, category)
            if not rows or name in special:
                continue
            if name == 'amyloid_unrecognized_pdb':
                if self.metadata_policy == 'preserve':
                    _set_cif_rows(out, category, rows)
                continue
            if name == 'audit_syntax':
                self._audit(category, 'invalidated',
                            'Source serializer column-layout hints do not describe the rewritten CIF.', rows)
                continue
            if name.startswith(self.INVALIDATED_PREFIXES):
                self._audit(category, "invalidated", "Describes the source assembly, frame, refinement, archive history or validation; recompute for this derivative.", rows)
                continue
            if name in self.CHAIN_CATEGORIES:
                mapped = []
                for group in self.groups:
                    group_atoms = group["output_rows"]
                    group_index = group["atom_index"]
                    for row in rows:
                        new = self._remap_row(row, group)
                        if new is None:
                            continue
                        if name in ('struct_ref_seq','struct_ref_seq_dif') and 'align_id' in new:
                            new_align = align_ids.get((group['index'], row['align_id']))
                            if new_align is None:
                                continue
                            new['align_id'] = new_align
                        if name == 'struct_site_gen':
                            new['site_id'] = site_ids[group['index'],row['site_id']]
                        if name in ('pdbx_nonpoly_scheme','pdbx_branch_scheme'):
                            seq = new.get('pdb_seq_num',new.get('auth_seq_num'))
                            component = new.get('mon_id',new.get('pdb_mon_id'))
                            if _present(seq) and not any(a['label_asym_id'] == new.get('asym_id') and
                                a['auth_seq_id'] == seq and a['label_comp_id'] == component and
                                _null_id(a.get('pdbx_PDB_ins_code')) == _null_id(new.get('pdb_ins_code')) for a in group_atoms):
                                continue
                        if name == "struct_conn":
                            if any(_present(row.get(p + "symmetry")) and row[p + "symmetry"] not in ("1_555", "1555") for p in ("ptnr1_", "ptnr2_")):
                                if row.get('conn_type_id') in ('disulf','covale','modres'):
                                    raise StructureEditError('A retained covalent connection uses crystal symmetry. Resolve the symmetry-related atoms explicitly before editing.')
                                self._audit(category + "symmetry", "invalidated", "Nonidentity crystal contacts cannot be copied to a finite fibril.", rows)
                                continue
                            if not all(group_index.exists(new, p) for p in ("ptnr1_", "ptnr2_")):
                                continue
                        if name == "struct_conf" and not all(group_index.exists(new, p) for p in ("beg_", "end_")):
                            continue
                        if name == "struct_site_gen" and not group_index.exists(new, ""):
                            continue
                        if name == "struct_mon_prot_cis" and not all(group_index.exists(new, p) for p in ("", "pdbx_")):
                            continue
                        for key in ("id", "pdbx_id"):
                            if key in new:
                                new[key] = str(len(mapped) + 1)
                        if name == 'struct_conf' and 'pdbx_PDB_helix_id' in new:
                            new['pdbx_PDB_helix_id'] = str(len(mapped)+1)
                        mapped.append(new)
                _set_cif_rows(out, category, mapped)
                self._audit(category, "remapped", "All partners must survive in the same rigid copy.")
            elif name in self.SAFE_CATEGORIES:
                kept = []
                for row in rows:
                    row = dict(row)
                    entity = row.get("entity_id", row.get("pdbx_entity_id", row.get("id") if name == "entity" else None))
                    if _present(entity) and name not in ("em_entity_assembly",) and entity not in live_entities:
                        continue
                    if name == "entity_poly" and "pdbx_strand_id" in row:
                        row["pdbx_strand_id"] = ",".join(dict.fromkeys(a["auth_asym_id"] for a in self.output_atoms if a.get("label_entity_id") == entity))
                    if name == "entity" and "pdbx_number_of_molecules" in row:
                        members = [a for a in self.output_atoms if a.get('label_entity_id') == entity and str(a['pdbx_PDB_model_num']) == self.models[0]]
                        poly_out = {g['labels'][old] for g in self.groups for old in self.poly_labels if old in g['labels']}
                        row['pdbx_number_of_molecules'] = str(len({a['label_asym_id'] if a['label_asym_id'] in poly_out else _residue_key(a) for a in members}))
                    if name == 'em_software' and _present(row.get('fitting_id')):
                        self._audit('_em_software.fitting_id', 'invalidated', 'Removed source-model fitting software rows with invalidated fitting records.', rows)
                        continue
                    if name == 'em_software':
                        row.pop('fitting_id', None)
                    kept.append(row)
                _set_cif_rows(out, category, kept)
                self._audit(category, "preserved/filtered", "Source descriptive/experimental metadata; entity membership/counts updated.")
            else:
                if self.metadata_policy == "error":
                    raise StructureEditError(f"No reviewed edit policy for {category}. Use metadata_policy='audit' to remove it with a full export report, or add a tested handler.")
                self._source_warning('unreviewed_cif_category',
                    (f'Unreviewed mmCIF category {category} is preserved unchanged when saving; '
                     'its references may describe the original structure.' if self.metadata_policy == 'preserve' else
                     f'Unreviewed mmCIF category {category} is archived in memory and omitted from the working copy; atom coordinates are retained.'),
                    category=category, rows=rows)
                if self.metadata_policy == 'preserve':
                    _set_cif_rows(out, category, rows)
                self._audit(category, 'preserved/unmapped' if self.metadata_policy == 'preserve' else 'removed',
                            'Unreviewed category; cannot promise its references remain valid.', rows)
        asym = []
        seen = set()
        for row in self.output_atoms:
            label = row["label_asym_id"]
            if label not in seen:
                asym.append({"id": label, "entity_id": row.get("label_entity_id")})
                seen.add(label)
        _set_cif_rows(out, "_struct_asym.", asym)
        self._audit('_struct_asym.', 'rebuilt', 'Independent component identifiers for every retained/generated copy.')
        self._sequence_scheme()
        self._sheets()
        sites = _cif_rows(out, "_struct_site_gen.")
        site_counts = Counter(r["site_id"] for r in sites)
        site_headers = []
        for (group_index, old_id), new_id in site_ids.items():
            group = self.groups[group_index]
            for row in _cif_rows(self.block, '_struct_site.'):
                if row['id'] != old_id or new_id not in site_counts:
                    continue
                mapped = self._remap_row(row, group)
                if mapped is None:
                    continue
                if _present(mapped.get('pdbx_auth_asym_id')) and not group['atom_index'].exists(mapped, 'pdbx_'):
                    continue
                site_headers.append(dict(mapped, id=new_id, pdbx_num_residues=str(site_counts[new_id])))
        retained_sites = {r['id'] for r in site_headers}
        _set_cif_rows(out, '_struct_site.', site_headers)
        _set_cif_rows(out, '_struct_site_gen.', [r for r in sites if r['site_id'] in retained_sites])
        self._audit('_struct_site.', 'remapped/filtered',
                    'Site-defining residue/ligand and member references must survive in the same copy.',
                    _cif_rows(self.block, '_struct_site.'))
        self.report['source_entry_id'] = self.block.find_value('_entry.id')
        for category in out.get_mmcif_category_names():
            if self.report['metadata'].get(category, {}).get('action') == 'preserved/unmapped':
                continue
            rows = _cif_rows(out, category)
            changed = False
            for row in rows:
                for key in row:
                    if key == 'entry_id' or (category == '_entry.' and key == 'id'):
                        row[key] = 'AMYLOID_MODIFIED'
                        changed = True
            if changed:
                _set_cif_rows(out, category, rows)

    def _sequence_scheme(self):
        if _cif_rows(self.output_block, '_pdbx_poly_seq_scheme.'):
            return
        sequence = defaultdict(list)
        for row in _cif_rows(self.output_block, '_entity_poly_seq.'):
            sequence[row['entity_id']].append(row)
        by_label = defaultdict(list)
        for row in self.output_atoms:
            if str(row['pdbx_PDB_model_num']) == self.models[0]:
                by_label[row['label_asym_id']].append(row)
        scheme = []
        for label, atoms in by_label.items():
            entity = atoms[0].get('label_entity_id')
            if entity not in sequence:
                continue
            observed = {}
            for atom in atoms:
                observed.setdefault((atom.get('label_seq_id'), atom['label_comp_id']), atom)
            for residue in sequence[entity]:
                atom = observed.get((residue['num'], residue['mon_id']))
                scheme.append({'asym_id':label, 'entity_id':entity, 'seq_id':residue['num'],
                    'mon_id':residue['mon_id'], 'ndb_seq_num':residue['num'],
                    'pdb_seq_num':atom.get('auth_seq_id') if atom else None,
                    'auth_seq_num':atom.get('auth_seq_id') if atom else None,
                    'pdb_mon_id':atom['label_comp_id'] if atom else None,
                    'auth_mon_id':atom.get('auth_comp_id', atom['label_comp_id']) if atom else None,
                    'pdb_strand_id':atoms[0]['auth_asym_id'],
                    'pdb_ins_code':(_null_id(atom.get('pdbx_PDB_ins_code')) or False) if atom else False,
                    'hetero':residue.get('hetero','n')})
        if scheme:
            _set_cif_rows(self.output_block, '_pdbx_poly_seq_scheme.', scheme)
            self._audit('_pdbx_poly_seq_scheme.', 'rebuilt', 'Full source sequence with observed author numbers/insertion codes; missing coordinates remain unknown.')

    def _sheets(self):
        ranges = _cif_rows(self.block, "_struct_sheet_range.")
        orders = _cif_rows(self.block, "_struct_sheet_order.")
        hbonds = _cif_rows(self.block, "_pdbx_struct_sheet_hbond.")
        sheets_out, ranges_out, order_out, hbond_out = [], [], [], []
        self.sheet_origins = {}
        for group in self.groups:
            atoms = group["atom_index"]
            sheet_ids = list(dict.fromkeys(r["sheet_id"] for r in ranges))
            for sheet in sheet_ids:
                surviving = []
                for r in ranges:
                    if r["sheet_id"] == sheet:
                        mapped = self._remap_row(r, group)
                        if mapped and all(self._endpoint_exists(mapped, atoms, p) for p in ("beg_", "end_")):
                            surviving.append(mapped)
                if not surviving:
                    continue
                new_sheet = str(len(sheets_out) + 1)
                self.sheet_origins[new_sheet] = sheet
                range_map = {r["id"]: str(i + 1) for i, r in enumerate(surviving)}
                sheets_out.append({"id": new_sheet, "number_strands": str(len(surviving))})
                for row in surviving:
                    ranges_out.append(dict(row, sheet_id=new_sheet, id=range_map[row["id"]]))
                for source, dest in ((orders, order_out), (hbonds, hbond_out)):
                    for row in source:
                        if row["sheet_id"] != sheet or any(row.get(k) not in range_map for k in ("range_id_1", "range_id_2")):
                            continue
                        mapped = self._remap_row(row, group)
                        if mapped is None:
                            continue
                        if source is hbonds and not all(self._endpoint_exists(mapped, atoms, p) for p in ("range_1_", "range_2_")):
                            continue
                        mapped.update(sheet_id=new_sheet, range_id_1=range_map[row["range_id_1"]], range_id_2=range_map[row["range_id_2"]])
                        dest.append(mapped)
        for cat, rows in (("struct_sheet", sheets_out), ("struct_sheet_range", ranges_out),
                          ("struct_sheet_order", order_out), ("pdbx_struct_sheet_hbond", hbond_out)):
            _set_cif_rows(self.output_block, "_" + cat + ".", rows)
            self._audit("_" + cat + ".", "remapped", "Surviving strand endpoints/registrations only; no new interlayer hydrogen bonds inferred.")

    def propagate_layer_sheets(self, sandwiches, instances, layer_step=1):
        # Extend only annotated sheet relationships; do not infer missing registrations.
        _, np = _structure_dependencies()
        locations = {c:(pf,l) for pf,s in enumerate(sandwiches) for l,c in enumerate(s)}
        ranges = _cif_rows(self.block, '_struct_sheet_range.')
        orders = _cif_rows(self.block, '_struct_sheet_order.')
        hbonds = _cif_rows(self.block, '_pdbx_struct_sheet_hbond.')
        first_atoms = [a for a in self.output_atoms if str(a['pdbx_PDB_model_num']) == self.models[0]]
        first_index = _AtomIndex(first_atoms)
        source_index = _AtomIndex([a for a in self.atoms if str(a['pdbx_PDB_model_num']) == self.models[0]])
        labels_by_chain = defaultdict(set)
        for a in first_atoms:
            if _present(a.get('label_seq_id')):
                labels_by_chain[a['auth_asym_id']].add(a['label_asym_id'])
        source_labels = defaultdict(set)
        for atom in self.atoms:
            if atom['label_asym_id'] in self.poly_labels:
                source_labels[atom['auth_asym_id']].add(atom['label_asym_id'])
        def mapped(row, chain_mapping):
            labels = {}
            for chain, destination in chain_mapping.items():
                if source_labels[chain]:
                    choices = labels_by_chain[destination]
                    if len(choices) != 1:
                        raise StructureEditError('Sheet propagation needs one polymer component per author chain.')
                    labels.update((label, next(iter(choices))) for label in source_labels[chain])
            return self._remap_row(row, {'chains':chain_mapping,'labels':labels})
        changed_sheets, new_components = set(), []
        copy_origins = {new:old for group in self.groups for old,new in group['chains'].items()}
        for sheet_id in dict.fromkeys(r['sheet_id'] for r in ranges):
            sr = [r for r in ranges if r['sheet_id'] == sheet_id]
            so = [r for r in orders if r['sheet_id'] == sheet_id]
            if not so or any(r.get('beg_auth_asym_id') != r.get('end_auth_asym_id') or r.get('beg_auth_asym_id') not in locations for r in sr):
                continue
            if len({locations[r['beg_auth_asym_id']][1] for r in sr}) < 2:
                continue
            # Match sheet intervals by ordinal position to allow differing strand endpoints.
            per_chain = defaultdict(list)
            for r in sr:
                per_chain[r['beg_auth_asym_id']].append(r)
            per_pf = defaultdict(set)
            for chain, chain_ranges in per_chain.items():
                per_pf[locations[chain][0]].add(len(chain_ranges))
            if any(len(counts) != 1 for counts in per_pf.values()):
                self.report['warnings'].append(f'Sheet {sheet_id} has inconsistent strand counts across layers; retained only directly mappable annotations.')
                continue
            slots = {}
            for chain, chain_ranges in per_chain.items():
                for slot,r in enumerate(sorted(chain_ranges,key=lambda r:(int(r['beg_auth_seq_id']),int(r['end_auth_seq_id'])))):
                    slots[r['id']] = slot
            nodes, source_nodes = {}, {}
            for row in sr:
                chain = row['beg_auth_asym_id']
                pf, layer = locations[chain]
                pattern = slots[row['id']]
                source_nodes[row['id']] = (pf, layer, pattern, chain)
                for (dest_pf, dest_layer), dest_chain in instances.items():
                    if dest_pf != pf or (dest_layer-layer) % layer_step or copy_origins.get(dest_chain) != chain:
                        continue
                    node = (pf, dest_layer, pattern)
                    out = mapped(row, {chain:dest_chain})
                    if out and all(first_index.exists(out, p) for p in ('beg_','end_')):
                        nodes.setdefault(node, out)
            edges, bonds = {}, {}
            for row in so:
                a, b = source_nodes.get(row['range_id_1']), source_nodes.get(row['range_id_2'])
                if a is None or b is None:
                    raise StructureEditError('Source sheet order has a missing range.')
                for target in list(nodes):
                    if target[0] != a[0] or target[2] != a[2]:
                        continue
                    shift = target[1]-a[1]
                    other = (b[0], b[1]+shift, b[2])
                    if shift % layer_step or other not in nodes:
                        continue
                    edge = (target, other)
                    edges[edge] = row
                    # Retain source registrations when copied sheet edges overlap.
                    for h in hbonds:
                        if h['sheet_id'] != sheet_id or h['range_id_1'] != row['range_id_1'] or h['range_id_2'] != row['range_id_2']:
                            continue
                        chain_map = {a[3]:instances[target[:2]], b[3]:instances[other[:2]]}
                        out = mapped(h,chain_map)
                        if not out:
                            continue
                        old_ends = [source_index.matches(h, p) for p in ('range_1_','range_2_')]
                        new_ends = [first_index.matches(out, p) for p in ('range_1_','range_2_')]
                        if any(not x for x in old_ends + new_ends):
                            continue
                        def length(pair):
                            return float(np.linalg.norm([float(pair[0][0][k])-float(pair[1][0][k]) for k in ('Cartn_x','Cartn_y','Cartn_z')]))
                        error = abs(length(old_ends)-length(new_ends))
                        if error <= .5 and (edge not in bonds or shift == 0):
                            bonds[edge] = out
            if not edges:
                continue
            changed_sheets.add(sheet_id)
            neighbors = {n:set() for n in nodes}
            for a,b in edges:
                neighbors[a].add(b); neighbors[b].add(a)
            remaining = set(nodes)
            while remaining:
                seed = min(remaining)
                component, pending = set(), [seed]
                while pending:
                    node = pending.pop()
                    if node in component:
                        continue
                    component.add(node); remaining.discard(node)
                    pending.extend(neighbors[node]-component)
                # A directed path determines strand ordering for PDB as well.
                incoming = {n:0 for n in component}
                for a,b in edges:
                    if a in component and b in component: incoming[b] += 1
                ordered, todo = [], sorted(n for n in component if incoming[n] == 0)
                while todo:
                    node = todo.pop(0); ordered.append(node)
                    for a,b in edges:
                        if a == node and b in component:
                            incoming[b] -= 1
                            if incoming[b] == 0: todo.append(b)
                if len(ordered) != len(component):
                    ordered = sorted(component)  # mmCIF can represent cyclic graphs.
                new_components.append((ordered,nodes,edges,bonds))
        if not changed_sheets:
            return self
        categories = ('_struct_sheet.','_struct_sheet_range.','_struct_sheet_order.','_pdbx_struct_sheet_hbond.')
        retained_ids = {new for new, old in self.sheet_origins.items() if old not in changed_sheets}
        output = {cat:[r for r in _cif_rows(self.output_block,cat) if r.get('sheet_id',r.get('id')) in retained_ids] for cat in categories}
        count = max([int(r['id']) for r in output['_struct_sheet.']] or [0])
        for ordered,nodes,edges,bonds in new_components:
            count += 1; new_id = str(count)
            ids = {node:str(i+1) for i,node in enumerate(ordered)}
            output['_struct_sheet.'].append({'id':new_id,'number_strands':str(len(ordered))})
            for node in ordered:
                output['_struct_sheet_range.'].append(dict(nodes[node],sheet_id=new_id,id=ids[node]))
            for (a,b), row in edges.items():
                if a not in ids or b not in ids: continue
                ids_update = dict(sheet_id=new_id,range_id_1=ids[a],range_id_2=ids[b])
                output['_struct_sheet_order.'].append(dict(row,**ids_update))
                if (a,b) in bonds:
                    output['_pdbx_struct_sheet_hbond.'].append(dict(bonds[a,b],**ids_update))
        for category, rows in output.items():
            _set_cif_rows(self.output_block,category,rows)
            self._audit(category,'remapped/extended','Translated explicitly annotated sheet-order patterns across layers; registration endpoints and distances checked.')
        self.report['periodic_sheets_extended'] = len(changed_sheets)
        self._validate_references()
        return self

    def _anisotrop(self):
        _, np = _structure_dependencies()
        rows = _cif_rows(self.block, "_atom_site_anisotrop.")
        output = []
        for group in self.groups:
            for row in rows:
                if row["id"] not in group["ids"]:
                    continue
                new = self._remap_row(row, group)
                if new is None:
                    raise StructureEditError("Anisotropic atom identity disagrees with atom_site.")
                new["id"] = group["ids"][row["id"]]
                for prefix in ("U", "B"):
                    keys = [f"{prefix}[{i}][{j}]" for i, j in ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))]
                    if all(_present(row.get(k)) for k in keys) and not np.array_equal(group["rotation"], np.eye(3)):
                        u11, u22, u33, u12, u13, u23 = (float(row[k]) for k in keys)
                        tensor = np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]])
                        rotated = group["rotation"] @ tensor @ group["rotation"].T
                        for key, (i, j) in zip(keys, ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
                            new[key] = f"{rotated[i, j]:.8f}"
                if not np.allclose(group["rotation"], np.eye(3)):
                    for key in list(new):
                        if "esd" in key.lower():
                            new[key] = None
                            self.report["warnings"].append("Anisotropic ESDs invalidated: full covariance is required to rotate uncertainties.")
                output.append(new)
        _set_cif_rows(self.output_block, "_atom_site_anisotrop.", output)

    def _conect_to_connections(self):
        rows = _cif_rows(self.output_block, "_struct_conn.")
        by_id = {r["id"]: r for r in self.output_atoms}
        first_index = _AtomIndex([a for a in self.output_atoms if str(a['pdbx_PDB_model_num']) == self.models[0]])
        existing_pairs = set()
        for row in rows:
            for a in first_index.matches(row, 'ptnr1_'):
                for b in first_index.matches(row, 'ptnr2_'):
                    existing_pairs.add(frozenset((a['id'], b['id'])))
        self.output_bonds = []
        for group in self.groups:
            for a, b, order in self.pdb_bonds:
                if a not in group["ids"] or b not in group["ids"]:
                    continue
                a, b = group["ids"][a], group["ids"][b]
                self.output_bonds.append((a, b, order))
                # struct_conn is model-independent; equivalent ensemble copies are represented once while PDB CONECT is written per model.
                if str(by_id[a]["pdbx_PDB_model_num"]) != self.models[0]:
                    continue
                # Merge duplicate bond annotations while preserving order and connection type.
                if frozenset((a, b)) in existing_pairs:
                    continue
                new = {"id": f"conect{len(rows) + 1}", "conn_type_id": "covale",
                       "pdbx_value_order": {1: "sing", 2: "doub", 3: "trip"}.get(order, "sing"),
                       "details": "Explicit source PDB CONECT; chemistry not inferred"}
                for prefix, atom in (("ptnr1_", by_id[a]), ("ptnr2_", by_id[b])):
                    for field in ("label_asym_id", "label_comp_id", "label_seq_id", "label_atom_id", "auth_asym_id", "auth_seq_id"):
                        new[prefix + field] = atom.get(field)
                    new["pdbx_" + prefix + "label_alt_id"] = atom.get("label_alt_id")
                    new["pdbx_" + prefix + "PDB_ins_code"] = atom.get("pdbx_PDB_ins_code")
                    new[prefix + "symmetry"] = "1_555"
                rows.append(new)
                # Unspecified altlocs remain wildcard endpoints, just as in the source matching rules, including subsequent CONECT records.
                for left in first_index.matches(new, 'ptnr1_'):
                    for right in first_index.matches(new, 'ptnr2_'):
                        existing_pairs.add(frozenset((left['id'], right['id'])))
        _set_cif_rows(self.output_block, "_struct_conn.", rows)
        if rows:
            types = {r["id"]: r for r in _cif_rows(self.output_block, "_struct_conn_type.")}
            for row in rows:
                types.setdefault(row["conn_type_id"], {"id": row["conn_type_id"]})
            _set_cif_rows(self.output_block, "_struct_conn_type.", list(types.values()))

    def propagate_layer_bonds(self, sandwiches, instances, layer_step=1, tolerance=0.20):
        # Propagate annotated covalent topology across repeat boundaries.
        _, np = _structure_dependencies()
        locations = {chain: (pf, layer) for pf, strand in enumerate(sandwiches) for layer, chain in enumerate(strand)}
        source_by_id = {r['id']: r for r in self.atoms}
        source_first = _AtomIndex([r for r in self.atoms if str(r['pdbx_PDB_model_num']) == self.models[0]])
        output_by_key = {_atom_key(r): r for r in self.output_atoms}
        rows = _cif_rows(self.output_block, '_struct_conn.')
        templates = []
        for row in _cif_rows(self.block, '_struct_conn.'):
            kind = row.get('conn_type_id', '')
            if kind not in ('disulf', 'covale', 'modres'):
                continue
            if any(_present(row.get(p + 'symmetry')) and row[p + 'symmetry'] not in ('1_555','1555') for p in ('ptnr1_', 'ptnr2_')):
                raise StructureEditError('A covalent connection uses crystal symmetry. Resolve its symmetry-related atoms explicitly before helical expansion.')
            endpoints = []
            for prefix in ('ptnr1_', 'ptnr2_'):
                matches = source_first.matches(row, prefix)
                endpoints.append(matches)
            if any(not m for m in endpoints):
                raise StructureEditError('Source covalent connection has a missing atom endpoint.')
            for a in endpoints[0]:
                for b in endpoints[1]:
                    aa, ab = _null_id(a.get('label_alt_id')), _null_id(b.get('label_alt_id'))
                    if aa and ab and aa != ab:
                        continue
                    templates.append((a, b, row))
        for a, b, order in self.pdb_bonds:
            if str(source_by_id[a]['pdbx_PDB_model_num']) == self.models[0]:
                templates.append((source_by_id[a], source_by_id[b],
                                  {'conn_type_id':'covale', 'pdbx_value_order':{1:'sing',2:'doub',3:'trip'}.get(order, 'sing')}))
        def endpoint_key(row, prefix):
            return tuple(_null_id(row.get(prefix + k)) for k in ('auth_asym_id','auth_seq_id','label_comp_id','label_atom_id')) + (
                _null_id(row.get('pdbx_' + prefix + 'PDB_ins_code')), _null_id(row.get('pdbx_' + prefix + 'label_alt_id')))
        def pair_key(row):
            return tuple(sorted((endpoint_key(row, 'ptnr1_'), endpoint_key(row, 'ptnr2_'))))
        seen = {pair_key(r) for r in rows}
        made = 0
        all_layers = sorted({layer for pf, layer in instances})
        for a, b, template in templates:
            if a['label_asym_id'] not in self.protein_labels or b['label_asym_id'] not in self.protein_labels:
                for group in self.groups:
                    if (a['id'] in group['ids']) != (b['id'] in group['ids']):
                        raise StructureEditError('Expansion would split a covalent connection involving a nonpolymer residue.')
                continue
            if a['auth_asym_id'] not in locations or b['auth_asym_id'] not in locations:
                continue  # Nonpolymer copies already follow their assigned owner.
            pfa, la = locations[a['auth_asym_id']]
            pfb, lb = locations[b['auth_asym_id']]
            if a['auth_asym_id'] == b['auth_asym_id']:
                # Whole-chain copies already carry their donor's internal bonds.
                # A bond annotated in a different layer must not bridge an
                # existing break in this donor chain (for example, 8FNZ).
                continue
            for dest_a in all_layers:
                shift = dest_a - la
                dest_b = lb + shift
                if shift % layer_step or (pfa, dest_a) not in instances or (pfb, dest_b) not in instances:
                    continue
                chain_a, chain_b = instances[pfa, dest_a], instances[pfb, dest_b]
                if chain_a in locations and chain_b in locations:
                    continue  # Existing links stay fixed, including unequal ends.
                # A link copied with both of its donor atoms already has the
                # donor's exact geometry. Another observed layer may have a
                # different bond length; it is not the reference for this copy.
                copied_pair = tuple(sorted((
                    (chain_a, _null_id(a['auth_seq_id']), a['label_comp_id'], a['label_atom_id'],
                     _null_id(a.get('pdbx_PDB_ins_code')), _null_id(a.get('label_alt_id'))),
                    (chain_b, _null_id(b['auth_seq_id']), b['label_comp_id'], b['label_atom_id'],
                     _null_id(b.get('pdbx_PDB_ins_code')), _null_id(b.get('label_alt_id'))))))
                if copied_pair in seen:
                    continue
                model_pairs = []
                for model in self.models:
                    ka, kb = list(_atom_key(a)), list(_atom_key(b))
                    ka[0], kb[0] = model, model
                    sa, sb = output_by_key.get(tuple(ka)), output_by_key.get(tuple(kb))
                    ka[1], kb[1] = chain_a, chain_b
                    oa, ob = output_by_key.get(tuple(ka)), output_by_key.get(tuple(kb))
                    if any(x is None for x in (sa, sb, oa, ob)):
                        raise StructureEditError('Cannot propagate covalent topology: a matching residue/atom/altloc is absent in an output layer/model.')
                    def distance(x, y):
                        return float(np.linalg.norm([float(x[k])-float(y[k]) for k in ('Cartn_x','Cartn_y','Cartn_z')]))
                    reference, actual = distance(sa, sb), distance(oa, ob)
                    if abs(actual - reference) > tolerance:
                        raise StructureEditError(f'Expanded covalent link {chain_a}/{a["auth_seq_id"]}/{a["label_atom_id"]} - {chain_b}/{b["auth_seq_id"]}/{b["label_atom_id"]} would be {actual:.3f} A (source {reference:.3f} A). Review layer assignment/twist/rise; no file written.')
                    model_pairs.append((oa, ob, actual))
                first_a, first_b, length = model_pairs[0]
                new = dict(template)
                new.update(id=f'layerbond{len(rows)+1}', pdbx_dist_value=f'{length:.3f}',
                           details='Covalent pattern translated from source layer/residue identities; all model bond lengths checked')
                for prefix, atom in (('ptnr1_', first_a), ('ptnr2_', first_b)):
                    for field in ('label_asym_id','label_comp_id','label_seq_id','label_atom_id','auth_asym_id','auth_seq_id'):
                        new[prefix+field] = atom.get(field)
                    new['pdbx_'+prefix+'PDB_ins_code'] = atom.get('pdbx_PDB_ins_code')
                    new['pdbx_'+prefix+'label_alt_id'] = atom.get('label_alt_id')
                    new[prefix+'symmetry'] = '1_555'
                if pair_key(new) in seen:
                    continue
                seen.add(pair_key(new))
                rows.append(new)
                order = {'sing':1,'doub':2,'trip':3}.get(new.get('pdbx_value_order'),1)
                for oa, ob, length in model_pairs:
                    self.output_bonds.append((oa['id'], ob['id'], order))
                made += 1
        _set_cif_rows(self.output_block, '_struct_conn.', rows)
        if rows:
            types = {r['id']: r for r in _cif_rows(self.output_block,'_struct_conn_type.')}
            for r in rows:
                types.setdefault(r['conn_type_id'], {'id':r['conn_type_id']})
            _set_cif_rows(self.output_block,'_struct_conn_type.', list(types.values()))
        self.report['inferred_covalent_links'] = made
        self.report['covalent_inference'] = 'Translation of annotated source patterns using layer offsets; 0.20 A maximum bond-length change in every model.'
        self._validate_references()
        return self

    def _validate_references(self):
        block = self.output_block
        asym = {r["id"]: r.get("entity_id") for r in _cif_rows(block, "_struct_asym.")}
        entities = {r["id"] for r in _cif_rows(block, "_entity.")}
        for row in self.output_atoms:
            if row["label_asym_id"] not in asym or (_present(row.get("label_entity_id")) and row["label_entity_id"] not in entities):
                raise StructureEditError("Dangling atom_site -> struct_asym/entity reference.")
        atom_ids = {r["id"] for r in self.output_atoms}
        for row in _cif_rows(block, "_atom_site_anisotrop."):
            if row["id"] not in atom_ids:
                raise StructureEditError("Dangling anisotropic atom reference.")
        auth = {r["auth_asym_id"] for r in self.output_atoms}
        for cat in block.get_mmcif_category_names():
            if self.report['metadata'].get(cat, {}).get('action') == 'preserved/unmapped':
                continue
            for row in _cif_rows(block, cat):
                for key, val in row.items():
                    if "asym_id" in key and _present(val):
                        known = auth if "auth" in key else asym
                        if val not in known:
                            raise StructureEditError(f"Dangling chain reference {cat}{key}={val}")
        for row in _cif_rows(block, "_struct_conn."):
            if not all(self.output_index.exists(row, p) for p in ("ptnr1_", "ptnr2_")):
                raise StructureEditError("Dangling struct_conn atom endpoint.")
        sulfur_partners = defaultdict(set)
        for row in _cif_rows(block, '_struct_conn.'):
            if row.get('conn_type_id') != 'disulf':
                continue
            partners = [self.output_index.matches(row, p) for p in ('ptnr1_','ptnr2_')]
            for a in partners[0]:
                for b in partners[1]:
                    if a['pdbx_PDB_model_num'] != b['pdbx_PDB_model_num']:
                        continue
                    aa, ab = _null_id(a.get('label_alt_id')), _null_id(b.get('label_alt_id'))
                    if aa and ab and aa != ab:
                        continue
                    # Alternate positions of the same partner count once.
                    sulfur_partners[_atom_key(a)].add(_atom_key(b)[:-1])
                    sulfur_partners[_atom_key(b)].add(_atom_key(a)[:-1])
        if any(len(partners)>1 for partners in sulfur_partners.values()):
            raise StructureEditError('A disulfide sulfur would have multiple residue partners. Review the alternating-layer pattern or provide explicit covalent topology; no file written.')

    def _pdb_text(self):
        gemmi, _ = _structure_dependencies()
        ranges_by_sheet = defaultdict(list)
        for row in _cif_rows(self.output_block, '_struct_sheet_range.'):
            ranges_by_sheet[row['sheet_id']].append(row['id'])
        for row in _cif_rows(self.output_block, '_struct_sheet_order.'):
            ids = ranges_by_sheet[row['sheet_id']]
            if row['range_id_1'] not in ids or row['range_id_2'] not in ids or ids.index(row['range_id_2']) != ids.index(row['range_id_1']) + 1:
                raise StructureEditError('This sheet graph is not a sequential PDB strand list; export mmCIF to preserve its topology.')
        max_chain = 1 if self.pdb_mode == "strict" else 2
        if any(len(r["auth_asym_id"]) > max_chain for r in self.output_atoms):
            raise StructureEditError(f"This PDB mode supports at most {max_chain} character(s) per chain ID; choose extended mode or mmCIF.")
        for row in self.output_atoms:
            for field, width in (("label_atom_id", 4), ("label_comp_id", 3)):
                if len(str(row[field])) > width:
                    raise StructureEditError(f"{field} overflows PDB; export mmCIF.")
            for field in ("label_alt_id", "pdbx_PDB_ins_code"):
                if len(_null_id(row.get(field))) > 1:
                    raise StructureEditError(f"{field} overflows PDB; export mmCIF.")
            try:
                seq = int(row["auth_seq_id"])
            except (ValueError, TypeError) as exc:
                raise StructureEditError("PDB requires integer author residue numbers; export mmCIF.") from exc
            if not -999 <= seq <= 9999:
                raise StructureEditError("PDB residue number overflow; export mmCIF.")
            for field in ("Cartn_x", "Cartn_y", "Cartn_z"):
                if len(f"{float(row[field]):8.3f}") != 8:
                    raise StructureEditError("PDB coordinate field overflow; export mmCIF.")
            for field in ("occupancy", "B_iso_or_equiv"):
                if not _present(row.get(field)) or len(f"{float(row[field]):6.2f}") != 6:
                    raise StructureEditError("PDB requires representable occupancy and B factor; export mmCIF.")
        st = gemmi.make_structure_from_block(self.output_block)
        st.setup_entities()
        st.assign_serial_numbers(numbered_ter=True)
        serial_by_key = {}
        for model in st:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        key = (str(model.num), chain.name, str(residue.seqid.num), residue.seqid.icode.strip(),
                               residue.name, atom.name, atom.altloc.replace("\x00", ""))
                        serial_by_key[key] = atom.serial
            if model.count_atom_sites() + len(model) > 99999:
                raise StructureEditError("PDB atom/TER serial capacity exceeded; export mmCIF.")
        st.clear_conect()
        by_id = {r["id"]: r for r in self.output_atoms}
        bond_serials = {}
        for a, b, order in self.output_bonds:
            pair = tuple(sorted((serial_by_key[_atom_key(by_id[a])], serial_by_key[_atom_key(by_id[b])])))
            bond_serials[pair] = max(order, bond_serials.get(pair, 0))
        for (a, b), order in bond_serials.items():
            st.add_conect(a, b, order)
        options = gemmi.PdbWriteOptions()
        options.preserve_serial = True
        options.numbered_ter = True
        options.conect_records = True
        options.use_linkr = False
        lines = st.make_pdb_string(options).splitlines()
        lines = [l for l in lines if l[:6].strip() not in ("MASTER", "END")]
        lines = self._pdb_supplement(lines)
        lines = [line[:62] + '    ' + line[66:] if line.startswith('HEADER') else line for line in lines]
        at = next((i for i, line in enumerate(lines) if line[:6].strip() in ('DBREF','SEQRES','MODRES','HET','HELIX','SHEET','CRYST1','ATOM','HETATM','MODEL')), len(lines))
        lines.insert(at, 'REMARK 999 DERIVED COORDINATES FROM AMYLOIDMODIFIER; SEE .AUDIT.JSON REPORT')
        for line in lines:
            if len(line) > 80 and line not in self.unrecognized_pdb_records:
                raise StructureEditError("PDB writer produced an overlong record; export mmCIF.")
            if line[:6].strip() in ("ATOM", "HETATM", "ANISOU", "TER"):
                if not line[6:11].strip().isdigit():
                    raise StructureEditError("PDB serial overflow; hybrid-36 is not enabled. Export mmCIF.")
        counts = Counter(line[:6].strip() for line in lines)
        values = (counts["REMARK"], 0, counts["HET"], counts["HELIX"], counts["SHEET"], 0,
                  counts["SITE"], sum(counts[r + str(i)] for r in ("ORIGX", "SCALE", "MTRIX") for i in (1, 2, 3)),
                  counts["ATOM"] + counts["HETATM"], counts["TER"], counts["CONECT"], counts["SEQRES"])
        if max(values) > 99999:
            raise StructureEditError("PDB MASTER count overflow; export mmCIF.")
        lines += ["MASTER    " + "".join(f"{v:5d}" for v in values), "END"]
        if max_chain == 2 and any(len(r["auth_asym_id"]) == 2 for r in self.output_atoms):
            self.report["warnings"].append("Uses the ChimeraX/Gemmi two-character PDB chain extension; not strict wwPDB v3.3.")
        return "\n".join(l if l in self.unrecognized_pdb_records else l.ljust(80) for l in lines) + "\n"

    def _pdb_supplement(self, generated):
        rebuilt = set("HEADER TITLE KEYWDS EXPDTA NUMMDL DBREF DBREF1 DBREF2 SEQRES MODRES HET HELIX SHEET SSBOND LINK CISPEP CRYST1 ORIGX1 ORIGX2 ORIGX3 SCALE1 SCALE2 SCALE3 MTRIX1 MTRIX2 MTRIX3 MODEL ATOM HETATM ANISOU TER ENDMDL CONECT MASTER END".split())
        descriptive = {"AUTHOR", "JRNL", "HETNAM", "HETSYN", "FORMUL", "MDLTYP"}
        invalidated = {"OBSLTE", "SPLIT", "CAVEAT", "REVDAT", "SPRSDE", "REMARK"}
        extra = []
        compound, source = self._pdb_compound_source()
        extra.extend(compound + source)
        extra.extend(self._pdb_sites())
        live_compounds = {r['label_comp_id'] for r in self.output_atoms}
        for rec in dict.fromkeys(l[:6].strip() for l in self.pdb_lines):
            rows = [l for l in self.pdb_lines if l[:6].strip() == rec]
            if rec in rebuilt or rec in ("COMPND", "SOURCE", "SITE") or not rec:
                continue
            if self.metadata_policy == 'preserve' and any(line[:6].strip() == rec for line in self.unrecognized_pdb_records):
                self._audit('PDB:' + rec, 'preserved/unmapped', 'Original unrecognized records retained verbatim.', rows)
                continue
            if rec in descriptive:
                if rec in ("HETNAM", "HETSYN"):
                    rows = [r for r in rows if r[11:14].strip() in live_compounds]
                elif rec == "FORMUL":
                    # Source copy-count multipliers cannot describe a new assembly.
                    self._audit("PDB:FORMUL", "invalidated", "Chemical formula copy counts must be regenerated from chemistry, not guessed.", rows)
                    continue
                extra.extend(rows)
                self._audit("PDB:" + rec, "preserved", "Source descriptive metadata.")
            elif rec == "SEQADV":
                for group in self.groups:
                    for row in rows:
                        old = row[15:17].strip()
                        if old in group["chains"]:
                            extra.append(row[:15] + f"{group['chains'][old]:>2}" + row[17:])
                self._audit("PDB:SEQADV", "remapped")
            elif rec in invalidated or self.metadata_policy in ("audit", "preserve"):
                self._audit("PDB:" + rec, "invalidated", "Unstructured source annotation cannot be asserted for the edited model; retained in report.", rows)
            else:
                raise StructureEditError(f"No reviewed policy for PDB record {rec}; use audit policy to archive it in the report.")
        # Header sections precede sequence/connectivity annotations and coordinates.
        order = "HEADER OBSLTE TITLE SPLIT CAVEAT COMPND SOURCE KEYWDS EXPDTA NUMMDL MDLTYP AUTHOR REVDAT SPRSDE JRNL REMARK DBREF DBREF1 DBREF2 SEQADV SEQRES MODRES HET HETNAM HETSYN FORMUL HELIX SHEET SSBOND LINK CISPEP SITE CRYST1 ORIGX1 ORIGX2 ORIGX3 SCALE1 SCALE2 SCALE3 MTRIX1 MTRIX2 MTRIX3".split()
        head, body = [], []
        for line in generated + extra:
            (head if line[:6].strip() in order else body).append(line)
        head.sort(key=lambda l: order.index(l[:6].strip()))
        if self.metadata_policy == 'preserve':
            head.extend(self.unrecognized_pdb_records)
            categories = {category: _cif_rows(self.output_block, category)
                          for category, audit in self.report['metadata'].items()
                          if category.startswith('_') and audit['action'] == 'preserved/unmapped'}
            if categories:
                head.extend(_cif_annotation_records(categories))
        return head + body

    def _pdb_compound_source(self):
        import textwrap
        def parse(rec):
            joined = ' '.join(l[10:80].strip() for l in self.pdb_lines if l[:6].strip() == rec)
            groups, current = {}, None
            for token in joined.split(';'):
                if not token.strip():
                    continue
                if ':' not in token:
                    raise StructureEditError(f'Cannot safely parse {rec} key/value continuation.')
                key, value = (s.strip() for s in token.split(':', 1))
                if key == 'MOL_ID':
                    current = value
                    groups[current] = []
                elif current is None:
                    raise StructureEditError(f'{rec} lacks a MOL_ID.')
                else:
                    groups[current].append((key, value))
            return groups
        compounds = parse('COMPND')
        sources = parse('SOURCE')
        live = {}
        for mol, fields in compounds.items():
            revised = []
            keep = True
            for key, value in fields:
                if key == 'CHAIN':
                    original = [v.strip() for v in value.split(',')]
                    mapped = list(dict.fromkeys(g['chains'][v] for g in self.groups for v in original if v in g['chains']))
                    keep = bool(mapped)
                    value = ', '.join(mapped)
                revised.append((key, value))
            if keep:
                live[mol] = revised
        def render(rec, groups):
            text = ' '.join('MOL_ID: ' + mol + '; ' + ' '.join(k + ': ' + v + ';' for k, v in fields)
                            for mol, fields in groups.items())
            return [f'{rec:<6}{str(i + 1) if i else "":>4}' + part
                    for i, part in enumerate(textwrap.wrap(text, 70, break_long_words=False, break_on_hyphens=False))]
        if compounds:
            self._audit('PDB:COMPND', 'remapped', 'Molecule chain lists expanded/filtered without changing source molecule identity.')
        if sources:
            self._audit('PDB:SOURCE', 'preserved/filtered', 'Retained MOL_IDs only.')
        return render('COMPND', live), render('SOURCE', {m: f for m, f in sources.items() if m in live})

    def _pdb_sites(self):
        sites = defaultdict(list)
        for line in self.pdb_lines:
            if line[:6].strip() == 'SITE':
                for start in (18, 29, 40, 51):
                    part = line.ljust(80)[start:start + 10]
                    if part.strip():
                        sites[line[11:14].strip()].append(part)
        output, site_number = [], 0
        for group in self.groups:
            live = {(r['auth_asym_id'], r['auth_seq_id'], _null_id(r.get('pdbx_PDB_ins_code')), r['label_comp_id'])
                    for r in group['rows']}
            for parts in sites.values():
                kept = []
                for part in parts:
                    chain, seq, ins, res = part[3:5].strip(), part[5:9].strip(), part[9].strip(), part[:3].strip()
                    if chain in group['chains'] and (chain, seq, ins, res) in live:
                        kept.append(part[:3] + f"{group['chains'][chain]:>2}" + part[5:])
                if not kept:
                    continue
                site_number += 1
                if site_number > 999 or len(kept) > 99:
                    raise StructureEditError('SITE field overflow; export mmCIF.')
                for i in range(0, len(kept), 4):
                    output.append(f'SITE   {i // 4 + 1:3d} {site_number:3d} {len(kept):2d} ' + ' '.join(kept[i:i + 4]))
        if sites:
            self._audit('PDB:SITE', 'remapped', 'Surviving complete residue references only; site IDs and counts rebuilt.')
        return output

    def _serialize_checked(self, format, output_name):
        gemmi, np = _structure_dependencies()
        if self.output_block is None:
            raise StructureEditError("Choose an edit before writing.")
        if format not in ('cif', 'mmcif', 'pdb'):
            raise StructureEditError("Specify PDB or mmCIF output format.")
        if format != 'pdb' and self.pdb_lines:
            # Keep untranslatable PDB annotations in the in-memory provenance report.
            coordinate_records = {'ATOM','HETATM','ANISOU','TER','MODEL','ENDMDL','CONECT','MASTER','END'}
            self.report['source_pdb_metadata'] = [l for l in self.pdb_lines if l[:6].strip() not in coordinate_records]
            self.report['warnings'].append('PDB-to-mmCIF conversion: unmapped free-text/source-only PDB metadata is archived in this report, not asserted as edited mmCIF categories.')
        text = self._pdb_text() if format == "pdb" else self.output_document.as_string()
        reopened = (gemmi.read_pdb_string(text) if format == 'pdb' else
                    gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block()))
        actual = []
        atom_properties = {}
        for model in reopened:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        key = (str(model.num), chain.name, str(residue.seqid.num), residue.seqid.icode.strip(),
                               residue.name, atom.name, atom.altloc.replace("\x00", ""))
                        actual.append((key,(atom.pos.x, atom.pos.y, atom.pos.z)))
                        atom_properties[key] = atom
        expected = {_atom_key(r): [float(r[f]) for f in ("Cartn_x", "Cartn_y", "Cartn_z")] for r in self.output_atoms}
        if len(actual) != len(expected) or {a[0] for a in actual} != set(expected):
            raise StructureEditError("Read-back atom identities/count differ from the requested edit.")
        tolerance = 0.00051 if format == "pdb" else 0.0000011
        if not np.allclose([xyz for key, xyz in actual], [expected[key] for key, xyz in actual], atol=tolerance, rtol=0):
            raise StructureEditError("Read-back coordinate precision check failed.")
        for row in self.output_atoms:
            atom = atom_properties[_atom_key(row)]
            for field, value in (('occupancy',atom.occ),('B_iso_or_equiv',atom.b_iso)):
                if _present(row.get(field)) and abs(float(row[field])-value) > (.0051 if format == 'pdb' else 1e-5) + abs(value)*1e-7:
                    raise StructureEditError('Read-back occupancy/B-factor check failed.')
            if _present(row.get('type_symbol')) and atom.element.name.upper() != str(row['type_symbol']).upper():
                raise StructureEditError('Read-back element identity differs.')
            if _present(row.get('pdbx_formal_charge')) and atom.charge != int(row['pdbx_formal_charge']):
                raise StructureEditError('Read-back formal charge differs.')
        back = reopened.make_mmcif_block(gemmi.MmcifOutputGroups(True,auth_all=True))
        def endpoint(row,prefix):
            return tuple(_null_id(row.get(prefix+k)) for k in ('auth_asym_id','auth_seq_id','label_comp_id','label_atom_id')) + (
                _null_id(row.get('pdbx_'+prefix+'PDB_ins_code',row.get(prefix+'PDB_ins_code'))),
                _null_id(row.get('pdbx_'+prefix+'label_alt_id')))
        def edge_set(block):
            return {tuple(sorted((endpoint(r,'ptnr1_'),endpoint(r,'ptnr2_'))))
                    for r in _cif_rows(block,'_struct_conn.') if r.get('conn_type_id') in ('disulf','covale','modres','metalc')}
        if not edge_set(self.output_block) <= edge_set(back):
            raise StructureEditError('Read-back lost an explicitly annotated covalent/metal connection.')
        def strand_set(block):
            return {(endpoint(r,'beg_'),endpoint(r,'end_')) for r in _cif_rows(block,'_struct_sheet_range.')}
        if strand_set(back) != strand_set(self.output_block):
            raise StructureEditError('Read-back sheet strand endpoints differ.')
        self.report.update(output=output_name, format=format, models=len(reopened), atoms=len(actual),
                           author_chains=len({r["auth_asym_id"] for r in self.output_atoms}),
                           output_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(), round_trip="passed",
                           round_trip_checks=['atom identities/counts','coordinates','occupancy/B','elements/charges','covalent/metal endpoints','sheet strand endpoints'])
        self.report["warnings"] = list(dict.fromkeys(self.report["warnings"]))
        return text

    def write(self, path, *, format=None, report_path=None, final_path=None):
        if isinstance(path, StructureBuffer):
            format = format or path.format
            text = self._serialize_checked(format, 'memory:' + path.name)
            # A failed check leaves both the previous coordinates and audit intact.
            path.text, path.report = text, deepcopy(self.report)
            path.format = 'pdb' if format == 'pdb' else 'cif'
            return self.report
        path = os.path.abspath(os.fspath(path))
        if format is None:
            suffix = os.path.splitext(path)[1].lower()
            format = 'cif' if suffix in ('.cif', '.mmcif') else 'pdb' if suffix in ('.pdb', '.ent') else None
        report_path = os.path.abspath(os.fspath(report_path or path + '.audit.json'))
        if report_path in (path, self.path):
            raise StructureEditError("Audit report must not overwrite a coordinate file.")
        output_name = os.path.abspath(os.fspath(final_path)) if final_path else path
        text = self._serialize_checked(format, output_name)
        suffix = '.pdb' if format == 'pdb' else '.cif'
        fd, temporary = tempfile.mkstemp(prefix='.amyloid-', suffix=suffix, dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            report_fd, report_tmp = tempfile.mkstemp(prefix='.amyloid-report-', dir=os.path.dirname(report_path))
            try:
                with os.fdopen(report_fd, 'w', encoding='utf-8') as handle:
                    json.dump(self.report, handle, indent=2, ensure_ascii=False)
                os.replace(report_tmp, report_path)
            finally:
                if os.path.exists(report_tmp):
                    os.unlink(report_tmp)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.report


def _fit_detection_core(p, q, fraction=.60):
    # Fit a stable core while allowing long disordered termini.
    import numpy as np
    n = len(p)
    keep = min(n, max(6, int(math.ceil(n * fraction))))
    seeds = [np.arange(n)]
    seeds.extend(np.arange(i, min(i + 12, n)) for i in range(0, n - 5, 6))
    best = None
    for mask in seeds:
        for _ in range(10):
            pc, qc = p[mask].mean(0), q[mask].mean(0)
            if np.linalg.matrix_rank(p[mask] - pc) < 2:
                break
            u, _, vt = np.linalg.svd((p[mask] - pc).T @ (q[mask] - qc))
            fix = np.eye(3)
            fix[2, 2] = np.linalg.det(vt.T @ u.T)
            rotation = vt.T @ fix @ u.T
            error = np.linalg.norm((p - pc) @ rotation.T + qc - q, axis=1)
            new = np.argsort(error, kind='stable')[:keep]
            score = float(np.sqrt(np.mean(error[new] ** 2)))
            if best is None or score < best['rmsd']:
                best = dict(rmsd=score, rotation=rotation, mask=new, error=error)
            if np.array_equal(np.sort(mask), np.sort(new)):
                break
            mask = new
        if best is not None and best['rmsd'] < .15:
            break
    if fraction == .60 and n >= 24 and (best is None or best['rmsd'] > 1.0):
        smaller = _fit_detection_core(p, q, .25)
        if smaller is not None and smaller['rmsd'] <= .5:
            inliers = np.flatnonzero(smaller['error'] <= .75)
            runs = np.split(inliers, np.where(np.diff(inliers) != 1)[0] + 1)
            if len(inliers) >= max(12, math.ceil(n * .25)) and max(map(len, runs)) >= 8:
                refined = _fit_detection_core(p[inliers], q[inliers], 1.)
                if refined is not None and refined['rmsd'] <= .75:
                    refined['mask'] = inliers[refined['mask']]
                    pc, qc = p[refined['mask']].mean(0), q[refined['mask']].mean(0)
                    refined['error'] = np.linalg.norm((p - pc) @ refined['rotation'].T + qc - q, axis=1)
                    return refined
    return best


def _short_unknown_chain(residues):
    return 4 <= len(residues) <= 12 and all(tuple(r.get('key', ()))[-1:] == ('UNK',) for r in residues)


def _compatible_residue_keys(a, b):
    if a == b:
        return True
    small, large = sorted((set(a), set(b)), key=len)
    return len(small) >= max(6, .75 * len(large)) and small.issubset(large)


def _sequence_coordinate_pairs(first, second):
    """Fallback for equivalent residues with different author numbering.

    Explicit mmCIF entity/sequence positions provide correspondence, while
    deposited author IDs remain untouched. Do not infer a sequence alignment
    from spatial proximity or quietly pair different residue types.
    """
    import numpy as np
    maps = []
    for residues in (first, second):
        keys = [r.get('sequence_key') for r in residues]
        if len(keys) < 6 or any(k is None or k[-1] == 'UNK' for k in keys):
            return None
        if len({k[:2] for k in keys}) != len(keys):
            return None
        if len({k[0] for k in keys}) != 1:
            return None
        maps.append(dict(zip(keys, residues)))
    a, b = maps
    if not _compatible_residue_keys(a.keys(), b.keys()):
        return None
    common = [k for k in a if k in b]
    return np.array([a[k]['coord'] for k in common]), np.array([b[k]['coord'] for k in common])


def _repeat_coordinate_pairs(first, second):
    # Align short unassigned peptides independently of author residue numbering.
    import numpy as np
    a = {r['key']: r['coord'] for r in first}
    b = {r['key']: r['coord'] for r in second}
    if set(a) == set(b):
        return np.array(list(a.values())), np.array([b[k] for k in a])
    if _compatible_residue_keys(a.keys(), b.keys()):
        common = [k for k in a if k in b]
        return np.array([a[k] for k in common]), np.array([b[k] for k in common])
    sequence_pairs = _sequence_coordinate_pairs(first, second)
    if sequence_pairs is not None:
        return sequence_pairs
    if not (_short_unknown_chain(first) and _short_unknown_chain(second)):
        raise StructureEditError('C-alpha residue identities differ between equivalent repeat positions; resolve alignment before automatic fitting.')
    p, q = np.array(list(a.values())), np.array(list(b.values()))
    choices = []
    required = max(4, math.ceil(.8 * min(len(p), len(q))))
    for offset in range(-len(q) + required, len(p) - required + 1):
        start_p, start_q = max(0, offset), max(0, -offset)
        size = min(len(p) - start_p, len(q) - start_q)
        pp, qq = p[start_p:start_p + size], q[start_q:start_q + size]
        delta = qq - pp
        score = float(np.mean(np.sum((delta - delta.mean(0)) ** 2, axis=1)) + .01 * np.mean(np.sum(delta ** 2, axis=1)))
        choices.append((score, -size, start_p, start_q))
    _, negative_size, i, j = min(choices)
    return p[i:i - negative_size], q[j:j - negative_size]


def _internal_rung_evidence(points, axis, spacing=4.8, max_rungs=6):
    import numpy as np
    n = len(points)
    counts = []
    for rung in range(1, max_rungs):
        pairs = 0
        participants = set()
        # Blocks avoid allocating an entire long-chain N x N x 3 tensor.
        for start in range(0, n, 128):
            delta = points[start:start + 128, None, :] - points[None, :, :]
            axial = delta @ axis
            transverse2 = np.maximum(0., np.sum(delta * delta, axis=2) - axial * axial)
            ii = np.arange(start, min(start + 128, n))[:, None]
            jj = np.arange(n)[None, :]
            hit = ((np.abs(axial - rung * spacing) <= .9) &
                   (transverse2 <= 2.5 ** 2) & (np.abs(ii - jj) >= 4))
            rows, cols = np.where(hit)
            pairs += len(rows)
            participants.update((rows + start).tolist())
            participants.update(cols.tolist())
        counts.append(dict(separation=rung, pairs=pairs, residues=len(participants)))
    supported = [r['separation'] + 1 for r in counts
                 if r['pairs'] >= 6 and r['residues'] >= max(12, math.ceil(n * .12))]
    return max([1] + supported), counts


def _antiparallel_contacts(p, q, axis=None):
    return _backbone_stack_contacts(p, q, axis, antiparallel=True)


def _backbone_stack_contacts(p, q, axis=None, antiparallel=True):
    import numpy as np
    direction = -1 if antiparallel else 1
    required = max(6, math.ceil(.25 * min(len(p), len(q))))
    if antiparallel and 6 <= min(len(p), len(q)) < 8:
        # A seven-residue strand has only five interior backbone tangents.
        # Keep the four-contact consecutive reverse-register requirement below
        # and require most of the available interior residues for short strands.
        required = max(4, math.ceil(.8 * (min(len(p), len(q)) - 2)))
    if min(len(p), len(q)) < required + 2:
        return None
    # Reject distant chains before allocating contact blocks.
    separation = np.maximum(0., np.maximum(p.min(0) - q.max(0), q.min(0) - p.max(0)))
    if np.linalg.norm(separation) > 7.2:
        return None

    def tangents(points):
        bonds = np.linalg.norm(np.diff(points, axis=0), axis=1)
        vectors = points[2:] - points[:-2]
        lengths = np.linalg.norm(vectors, axis=1)
        valid = ((bonds[:-1] >= 2.8) & (bonds[:-1] <= 4.3) &
                 (bonds[1:] >= 2.8) & (bonds[1:] <= 4.3) & (lengths >= 5.4))
        return vectors / np.maximum(lengths[:, None], 1e-12), valid

    u, valid_p = tangents(p)
    v, valid_q = tangents(q)
    contacts = []
    for start in range(0, len(u), 128):
        delta = q[None, 1:-1] - p[1:-1][start:start + 128, None]
        distance2 = np.sum(delta * delta, axis=2)
        hit = ((direction * (u[start:start + 128] @ v.T) > .6) &
               valid_p[start:start + 128, None] & valid_q[None, :] &
               (distance2 >= 3.5 ** 2) & (distance2 <= 7.2 ** 2))
        cost = distance2
        if axis is not None:
            axial = delta @ axis
            cost = np.maximum(0., distance2 - axial * axial)
            hit &= (np.abs(axial) >= 3.5) & (np.abs(axial) <= 6.5) & (cost <= 3.0 ** 2)
        ii, jj = np.where(hit)
        contacts.extend((float(cost[i, j]), start + int(i), int(j), delta[i, j])
                        for i, j in zip(ii, jj))
    # One residue cannot inflate the support by contacting multiple partners.
    pairs, used_p, used_q = [], set(), set()
    for cost, i, j, vector in sorted(contacts, key=lambda c: c[:3]):
        if i not in used_p and j not in used_q:
            pairs.append((i, j, vector))
            used_p.add(i)
            used_q.add(j)
    if len(pairs) < required:
        return None
    registers = {(i, j) for i, j, _ in pairs}
    if not any(all((i + step, j + direction * step) in registers for step in range(4))
               for i, j in registers):
        return None
    vectors = np.array([d for _, _, d in pairs])
    if axis is None:
        strand = np.array([u[i] + direction * v[j] for i, j, _ in pairs])
        strand /= np.linalg.norm(strand, axis=1)[:, None]
        transverse = vectors - np.sum(vectors * strand, axis=1)[:, None] * strand
        direction = transverse.mean(0)
        norm = float(np.linalg.norm(direction))
        if not 3.5 <= norm <= 6.5:
            return None
        axis = direction / norm
        # Recheck contacts against the inferred direction and common spacing.
        return _backbone_stack_contacts(p, q, axis, antiparallel)
    axial = vectors @ axis
    sign = 1 if np.median(axial) > 0 else -1
    if np.count_nonzero(sign * axial > 0) < .85 * len(pairs):
        return None
    rise = float(np.mean(sign * axial))
    if not 4.0 <= rise <= 5.7:
        return None
    # Contacts within a long, overlapping multi-rung chain must not merge stacks.
    center_shift = q.mean(0) - p.mean(0)
    center_rise = float(center_shift @ axis) * sign
    # Transverse alignment is checked on the contact residues above. A whole-chain lateral centroid shifts when termini are missing or flexible.
    if not 2.5 <= center_rise <= 7.5:
        return None
    return dict(vector=axis * sign * rise, distance=rise, rise=rise,
                support=len(pairs), contact_pairs=[(i + 1, j + 1) for i, j, _ in pairs],
                rmsd=None, orientation='antiparallel' if antiparallel else 'parallel',
                rungs=1, gap=1, spacing=rise)


def _shared_detection_helix(candidates, axis):
    """Recover wide helical repeats using independent, already aligned pairs.

    A centroid follows a circular arc around a screw axis. Its transverse
    displacement is not a stacking error, but a pair's own rigid fit alone
    cannot distinguish that arc from a displaced chain. Require several pairs
    to agree on one transform before relaxing the transverse-distance gate.
    """
    import numpy as np
    anchors = []
    for e in candidates:
        dz = float(e['vector'] @ axis)
        lateral = np.linalg.norm(e['vector'] - dz * axis)
        if e['rmsd'] > .35 or not 4.0 <= abs(dz) <= 5.7 or lateral > 2.0:
            continue
        p, q = e['core_p'], e['core_q']
        rotation = e['rotation']
        if dz < 0:
            p, q, rotation = q, p, rotation.T
        anchors.append((e, p, q, rotation, q.mean(0) - rotation @ p.mean(0)))
    if len(anchors) < 3:
        return None
    centers_p = np.array([a[1].mean(0) for a in anchors])
    centers_q = np.array([a[2].mean(0) for a in anchors])
    rotations = np.array([a[3] for a in anchors])
    best = []
    for _, p, q, rotation, translation in anchors:
        # Both rotation and its action at each pair's location must agree.
        center_error = np.linalg.norm(centers_p @ rotation.T + translation - centers_q, axis=1)
        cos_angles = (np.einsum('nij,ij->n', rotations, rotation) - 1) / 2
        support = np.flatnonzero((center_error <= .35) &
                                 (cos_angles >= math.cos(math.radians(.5))))
        if len(support) > len(best):
            best = support
    if len(best) < 3:
        return None
    members = [anchors[i] for i in best]
    if len({c for e, *_ in members for c in (e['a'], e['b'])}) < 4:
        return None
    p = np.concatenate([a[1] for a in members])
    q = np.concatenate([a[2] for a in members])
    fit = _fit_detection_core(p, q, 1.)
    if fit is None or fit['rmsd'] > .35:
        return None
    rotation = fit['rotation']
    translation = q.mean(0) - rotation @ p.mean(0)
    skew = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / 2
    sine = np.linalg.norm(skew)
    if sine < math.sin(math.radians(.1)):
        return None  # Near-zero twist does not determine a reliable screw axis.
    screw_axis = skew / sine
    if screw_axis @ axis < 0:
        screw_axis = -screw_axis
    if screw_axis @ axis < math.cos(math.radians(15)):
        return None
    rise = float(translation @ screw_axis)
    if not 4.0 <= rise <= 5.7:
        return None
    supported, recovered = set(), False
    for i, e in enumerate(candidates):
        p, q = e['core_p'], e['core_q']
        if e['vector'] @ screw_axis < 0:
            p, q = q, p
        error = np.linalg.norm(p @ rotation.T + translation - q, axis=1)
        if np.sqrt(np.mean(error ** 2)) > .5:
            continue
        supported.add(i)
        dz = float(e['vector'] @ axis)
        lateral = np.linalg.norm(e['vector'] - dz * axis)
        recovered |= lateral > max(2.0, .30 * abs(dz))
    # Leave the established detector untouched unless the shared transform
    # actually explains an otherwise rejected transverse displacement.
    return (screw_axis, supported) if recovered else None


def detect_layers(chains, axis_hint=None):
    # Detect whole chain units; a chain may span multiple physical layers.
    import numpy as np
    ids = sorted(chains)
    if not ids:
        raise StructureEditError('No C-alpha detected; detection failed.')
    maps, points = {}, {}
    for cid in ids:
        maps[cid] = {}
        for residue in chains[cid]:
            key = tuple(residue.get('key', (residue['id'],)))
            if key in maps[cid]:
                raise StructureEditError('Duplicate C-alpha residue identity in chain ' + cid)
            coord = np.asarray(residue['coord'], dtype=float)
            if coord.shape != (3,) or not np.all(np.isfinite(coord)):
                raise StructureEditError('Invalid C-alpha coordinates in chain ' + cid)
            maps[cid][key] = coord
        points[cid] = np.array(list(maps[cid].values()))
        if not len(points[cid]):
            raise StructureEditError('Empty C-alpha chain ' + cid)

    candidates = []
    contact_candidates = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            anti = _antiparallel_contacts(points[a], points[b])
            if anti is not None:
                contact_candidates.append(dict(anti, a=a, b=b))
            common = [k for k in maps[a] if k in maps[b]]
            # Use local backbone contacts when neighboring proteins have different sequences.
            sequence_a = tuple(k[-1] for k in maps[a] if len(k) >= 3)
            sequence_b = tuple(k[-1] for k in maps[b] if len(k) >= 3)
            if sequence_a != sequence_b:
                contact = _backbone_stack_contacts(points[a], points[b], antiparallel=False)
                if contact is not None:
                    contact_candidates.append(dict(contact, a=a, b=b))
            # Short assigned peptides use every residue in the same repeat fit.
            # They must match completely; unlike a long chain there is no spare
            # core support to trim away mismatches or flexible residues.
            short_assigned = (3 <= len(maps[a]) < 6 and set(maps[a]) == set(maps[b])
                              and all(len(k) >= 3 and k[-1] != 'UNK' for k in maps[a]))
            if not short_assigned and len(common) < max(6, math.ceil(.5 * min(len(maps[a]), len(maps[b])))):
                pairs = _sequence_coordinate_pairs(chains[a], chains[b])
                if pairs is None:
                    continue
                p, q = pairs
            else:
                p = np.array([maps[a][k] for k in common])
                q = np.array([maps[b][k] for k in common])
            if short_assigned:
                delta = q - p
                residual = float(np.sqrt(np.mean(np.sum((delta - delta.mean(0)) ** 2, axis=1))))
                if (residual > .9 or np.linalg.matrix_rank(p - p.mean(0)) < 2
                        or np.linalg.matrix_rank(q - q.mean(0)) < 2):
                    continue
            lengths = np.linalg.norm(q - p, axis=1)
            if np.percentile(lengths, 25) > 36 or np.max(lengths) < 3.5:
                continue
            fit = _fit_detection_core(p, q)
            if fit is None or fit['rmsd'] > 1.0:
                continue
            angle = math.degrees(math.acos(float(np.clip((np.trace(fit['rotation']) - 1) / 2, -1, 1))))
            if angle > 25:
                continue
            mask = fit['mask']
            vector = (q - p)[mask].mean(0)
            distance = float(np.linalg.norm(vector))
            if not 3.8 <= distance <= 34.2:
                continue
            # Long repeats need internal rung evidence. Ordinary missing-chain gaps may be at most three rungs; they never increase rungs/unit.
            if distance > 17.1:
                ka, _ = _internal_rung_evidence(points[a], vector / distance)
                kb, _ = _internal_rung_evidence(points[b], vector / distance)
                if distance > 5.7 * min(ka, kb):
                    continue
            candidates.append(dict(a=a, b=b, vector=vector, distance=distance,
                                   rmsd=fit['rmsd'], support=len(mask),
                                   orientation='parallel',
                                   rotation=fit['rotation'], core_p=p[mask], core_q=q[mask],
                                   center_a=p[mask].mean(0), center_b=q[mask].mean(0)))

    warnings = []
    hint = None
    if axis_hint is not None:
        hint = np.asarray(axis_hint, dtype=float)
        if hint.shape != (3,) or not np.all(np.isfinite(hint)) or np.linalg.norm(hint) < 1e-8:
            raise StructureEditError('An axis hint must be a finite nonzero three-vector.')
        hint = hint / np.linalg.norm(hint)
    axis = None
    edges = []
    rung_evidence = {}
    multiplicity = {c: 1 for c in ids}
    shared_helix = None
    axis_candidates = candidates or contact_candidates
    if axis_candidates:
        # Estimate direction from nearest repeats, independent of sign and lateral separation.
        seeds = {min((i for i, e in enumerate(axis_candidates) if c in (e['a'], e['b'])),
                     key=lambda i: (axis_candidates[i]['distance'], axis_candidates[i]['rmsd'] or 0.))
                 for c in ids if any(c in (e['a'], e['b']) for e in axis_candidates)}
        directions = np.array([axis_candidates[i]['vector'] / axis_candidates[i]['distance'] for i in sorted(seeds)])
        # Choose the strongest angular cluster before estimating its direction.
        agreement = np.abs(directions @ directions.T) >= math.cos(math.radians(25))
        selected = agreement[np.argmax(agreement.sum(1))]
        _, _, vt = np.linalg.svd(directions[selected], full_matrices=False)
        axis = vt[0]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        shared_helix = _shared_detection_helix(candidates, axis)
        if shared_helix is not None:
            axis = shared_helix[0]
        for c in ids:
            multiplicity[c], rung_evidence[c] = _internal_rung_evidence(points[c], axis)
        for index, e in enumerate(candidates):
            dz = float(e['vector'] @ axis)
            lateral = float(np.linalg.norm(e['vector'] - dz * axis))
            k = min(multiplicity[e['a']], multiplicity[e['b']])
            # A long linker may create extra contacts in one chain; multiplicity must also agree with the observed repeat, on both sides.
            repeat_rungs = max(1, round(abs(dz) / 4.8))
            k = min(k, repeat_rungs)
            gap = max(1, round(abs(dz) / (4.8 * k)))
            spacing = abs(dz) / (gap * k)
            helical_neighbor = shared_helix is not None and index in shared_helix[1]
            if (gap > 3 or not 4.0 <= spacing <= 5.7 or
                    (lateral > max(2.0, .30 * abs(dz)) and not helical_neighbor)):
                continue
            if dz < 0:
                e = dict(e, a=e['b'], b=e['a'], center_a=e['center_b'], center_b=e['center_a'])
            edges.append(dict(e, rise=abs(dz), rungs=k, gap=gap, spacing=spacing))

        for candidate in contact_candidates:
            a, b = candidate['a'], candidate['b']
            if any({a, b} == {e['a'], e['b']} and e['gap'] == 1 for e in edges):
                continue  # Keep an already supported rigid repeat unchanged.
            if multiplicity[a] != 1 or multiplicity[b] != 1:
                continue
            e = _backbone_stack_contacts(points[a], points[b], axis,
                                         antiparallel=candidate['orientation'] == 'antiparallel')
            if e is not None:
                if float(e['vector'] @ axis) < 0:
                    a, b = b, a
                    e['contact_pairs'] = [(j, i) for i, j in e['contact_pairs']]
                edges.append(dict(e, a=a, b=b))

        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if not (_short_unknown_chain(chains[a]) and _short_unknown_chain(chains[b])):
                    continue
                if any({a, b} == {e['a'], e['b']} for e in edges):
                    continue
                p, q = _repeat_coordinate_pairs(chains[a], chains[b])
                delta = q - p
                vector = delta.mean(0)
                dz = float(vector @ axis)
                lateral = float(np.linalg.norm(vector - dz * axis))
                residual = float(np.sqrt(np.mean(np.sum((delta - vector) ** 2, axis=1))))
                if not 4.0 <= abs(dz) <= 5.7 or lateral > 2.0 or residual > .9:
                    continue
                low, high = (a, b) if dz > 0 else (b, a)
                edges.append(dict(a=low, b=high, rise=abs(dz), rungs=1, gap=1,
                                  spacing=abs(dz), rmsd=residual, support=len(p), orientation='parallel'))

    # Use the same disjoint paths for grouping and ordering chains.
    up, down, chosen = {}, {}, []
    for e in sorted(edges, key=lambda e: (e['gap'], e['rise'], e['rmsd'] or 0., e['a'], e['b'])):
        a, b = e['a'], e['b']
        if a in up or b in down:
            continue
        cursor = b
        while cursor in up and cursor != a:
            cursor = up[cursor]['b']
        if cursor == a:
            continue
        up[a] = e
        down[b] = e
        chosen.append(e)

    # A flexible terminal chain can fail the rigid-core cutoff while still
    # forming a well-supported beta-sheet with an established stack. Join only
    # unambiguous path ends; never replace accepted edges or use these weaker
    # fits to estimate the axis. This also recognizes an expanded flexible end.
    if chosen and axis is not None:
        paths = []
        for start in ids:
            if start in down:
                continue
            path = [start]
            while path[-1] in up:
                path.append(up[path[-1]]['b'])
            if all(up[c]['rungs'] == 1 and up[c]['gap'] == 1 and
                   up[c]['orientation'] == 'parallel' for c in path[:-1]):
                paths.append(path)
        recoveries = []
        for first in paths if len({len(p) for p in paths}) > 1 else []:
            for second in paths:
                if first is second or max(len(first), len(second)) < 3:
                    continue
                a, b = first[-1], second[0]
                if set(maps[a]) != set(maps[b]):
                    continue
                p, q = _repeat_coordinate_pairs(chains[a], chains[b])
                if np.percentile(np.linalg.norm(q - p, axis=1), 25) > 12:
                    continue
                fit = _fit_detection_core(p, q)
                if fit is None or fit['rmsd'] > 1.5:
                    continue
                angle = math.degrees(math.acos(float(np.clip((np.trace(fit['rotation']) - 1) / 2, -1, 1))))
                vector = (q - p)[fit['mask']].mean(0)
                dz = float(vector @ axis)
                if angle > 25 or not 4.0 <= dz <= 5.7 or np.linalg.norm(vector - dz * axis) > 2.0:
                    continue
                contact = _backbone_stack_contacts(points[a], points[b], axis, antiparallel=False)
                if contact is None:
                    continue
                recoveries.append(dict(contact, a=a, b=b, rmsd=fit['rmsd']))
        counts = Counter((side, e[side]) for e in recoveries for side in ('a', 'b'))
        for e in recoveries:
            if counts['a', e['a']] != 1 or counts['b', e['b']] != 1:
                continue
            up[e['a']] = e
            down[e['b']] = e
            chosen.append(e)
            warnings.append('Flexible stack boundary ' + e['a'] + ' / ' + e['b'] +
                            ' joined by supported backbone contacts along the established axis.')

    if not chosen and hint is not None:
        axis = hint
        for c in ids:
            multiplicity[c], rung_evidence[c] = _internal_rung_evidence(points[c], axis)
    tracks = []
    for start in ids:
        if start in down:
            continue
        path, unit_positions = [start], [0]
        current = start
        while current in up:
            e = up[current]
            current = e['b']
            path.append(current)
            unit_positions.append(unit_positions[-1] + e['gap'])
        path_edges = [up[c] for c in path[:-1]]
        k = int(round(float(np.median([e['rungs'] for e in path_edges])))) if path_edges else (multiplicity[start] if hint is not None and not chosen else 1)
        rise = float(np.median([e['rise'] / e['gap'] for e in path_edges])) if path_edges else None
        if path_edges and any(e['rungs'] != k for e in path_edges):
            warnings.append('Inconsistent internal layer multiplicity in chain stack ' + ', '.join(path))
        orientations = {e['orientation'] for e in path_edges if e['gap'] == 1}
        orientation = ('mixed' if len(orientations) > 1 else next(iter(orientations))) if orientations else 'undetermined'
        tracks.append(dict(chains=path, units=len(path), layers_per_unit=k, orientation=orientation,
                           layers=len(path) * k, unit_positions=unit_positions,
                           axial_layer_span=(unit_positions[-1] + 1) * k,
                           unit_rise=rise, layer_spacing=rise / k if rise else None,
                           complete=unit_positions == list(range(len(path)))))

    tracks.sort(key=lambda t: (-t['units'], tuple(t['chains'])))
    if not chosen and hint is None:
        axis = None
        warnings.append('No inter-unit repeat detected: axis and layers per unit are undetermined; singleton chains are shown provisionally as one layer each.')
    elif not chosen:
        warnings.append('Axis supplied from the previously detected parent stack; internal contacts were rechecked. This single unit does not independently determine a repeat axis.')
    elif any(t['units'] == 1 for t in tracks):
        warnings.append('Some chains have no supported stacking neighbor; their layer multiplicity is undetermined.')
    if any(not t['complete'] for t in tracks):
        warnings.append('Missing chain units detected; occupied layers and axial layer span differ.')
    uniform = len({t['layers_per_unit'] for t in tracks}) == 1
    if not uniform:
        warnings.append('Protofilaments have different layers per chain unit.')
    # Keep every input chain exactly once, even in a partial/ambiguous assembly.
    assert sorted(c for t in tracks for c in t['chains']) == ids
    orientations = {t['orientation'] for t in tracks} - {'undetermined'}
    orientation = ('mixed' if len(orientations) > 1 else next(iter(orientations))) if orientations else 'undetermined'
    return dict(version='3.2', protofilaments=tracks, orientation=orientation,
                antiparallel=any(e['orientation'] == 'antiparallel' for e in chosen),
                sandwiches=[t['chains'] for t in tracks],
                axis=axis.tolist() if axis is not None else None,
                axis_source=('shared helical repeats' if shared_helix is not None else 'neighbor repeats' if candidates else
                             ('antiparallel backbone contacts' if orientation == 'antiparallel' else 'backbone contacts')) if chosen else ('supplied parent axis' if hint is not None else 'undetermined'),
                detected_units=max(t['units'] for t in tracks),
                detected_layers=max(t['layers'] for t in tracks),
                layers_per_unit=tracks[0]['layers_per_unit'] if uniform else None,
                complete=uniform and all(t['complete'] for t in tracks) and
                         len({t['units'] for t in tracks}) == 1,
                warnings=warnings, rung_evidence=rung_evidence,
                accepted_edges=[{k: e[k] for k in ('a', 'b', 'rise', 'rungs', 'gap', 'rmsd', 'support', 'orientation', 'contact_pairs') if k in e} for e in chosen])


def _expansion_repeat_pattern(chains, sandwiches):
    import numpy as np
    tags, contact_vectors, pure_anti = [], [], False
    for stack in sandwiches:
        points = [np.array([r['coord'] for r in chains[c]]) for c in stack]
        polarity, flips, stack_tags = 1, [], []
        for i, cid in enumerate(stack):
            if i:
                contact = _antiparallel_contacts(points[i - 1], points[i])
                flips.append(contact is not None)
                if contact is not None:
                    polarity *= -1
                    contact_vectors.append(contact['vector'])
            identity = ('short_unknown_peptide' if _short_unknown_chain(chains[cid]) else
                        frozenset(tuple(r.get('key', (r['id'],))) for r in chains[cid]))
            stack_tags.append((identity, polarity))
        pure_anti |= bool(flips) and all(flips)
        tags.append(stack_tags)
    depth = len(sandwiches[0])
    def same_identity(a, b, first, second):
        if a == b:
            return True
        if isinstance(a, frozenset) and isinstance(b, frozenset):
            # Unmodelled residues do not change protein identity. Require a
            # substantial matching sequence and no contradictory residue keys.
            if _compatible_residue_keys(a, b):
                return True
        return _sequence_coordinate_pairs(chains[first], chains[second]) is not None
    def compatible(period):
        return (not (pure_anti and period % 2) and
                all(all(t[i][1] == t[i + period][1] and same_identity(t[i][0], t[i + period][0], stack[i], stack[i + period])
                        for i in range(len(t) - period)) for t, stack in zip(tags, sandwiches)))
    # Preserve a previously usable exact-identity repeat. Partial coverage is
    # a fallback when exact matching leaves no observed repeat to fit, not a
    # reason to change a working structure's default templates or transform.
    strict_period = next((p for p in range(1, depth + 1)
                          if not (pure_anti and p % 2) and
                          all(all(t[i] == t[i + p] for i in range(len(t) - p)) for t in tags)), depth + 1)
    period = (strict_period if strict_period < depth else
              next((p for p in range(1, depth + 1) if compatible(p)), depth + 1))
    axis = None
    if contact_vectors:
        _, _, vt = np.linalg.svd(np.array(contact_vectors), full_matrices=False)
        axis = vt[0]
        if np.dot(axis, np.mean(contact_vectors, axis=0)) < 0:
            axis = -axis
    return dict(period=period, compatible=compatible, axis=axis,
                antiparallel=bool(contact_vectors))


def _single_unit_axis(points):
    """Estimate a layer normal from its C-alpha covariance, not a repeat fit.

    The two largest principal spreads define the broadest PCA projection;
    their normal is the smallest-eigenvalue eigenvector. Eigenvalues are
    coordinate variances (A^2), not literal molecular projected areas.
    """
    import numpy as np
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.all(np.isfinite(points)):
        raise StructureEditError('Estimating a single-unit axis needs at least three finite C-alpha coordinates.')
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    tolerance = max(1e-12, float(values[-1]) * 1e-6)
    if values[1] <= tolerance or values[1] - values[0] <= tolerance:
        raise StructureEditError('The single-unit shape has no unique plane normal. '
                                 'Align the structure with Z and uncheck Compute Helical Axis.')
    axis = vectors[:, 0]
    # A single layer cannot establish polarity; choose a reproducible sign.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis, np.maximum(values, 0.)


def terminal_completion_supported(detection):
    """Topology eligibility only; the expansion also validates the rigid fit."""
    tracks = (detection or {}).get('protofilaments', [])
    sizes = [t['units'] for t in tracks]
    if not sizes or not all(t['complete'] and t['layers_per_unit'] == 1 for t in tracks):
        return False
    missing = max(sizes) - min(sizes)
    established = (missing == 1 and min(sizes) >= 3 and
                   all(t['orientation'] in ('parallel', 'antiparallel', 'mixed') for t in tracks))
    short_parallel = (1 <= missing <= 2 and min(sizes) >= 2 and
                      all(t['orientation'] == 'parallel' for t in tracks))
    return established or short_parallel


def _edit_clone(editor):
    # Editing replaces output state; source tables are read-only. Keep reports
    # independent so any failed staged operation leaves the caller untouched.
    clone = object.__new__(StructureEditor)
    clone.__dict__ = editor.__dict__.copy()
    clone.report = deepcopy(editor.report)
    clone.report['chain_copies'] = []
    return clone


def terminal_completion_messages(report):
    completion = report.get('terminal_completion')
    if not completion:
        return []
    lines = [f"Modeled terminal completion: added {completion['added_chain_units']} chain units to make every protofilament {completion['completed_units']} layers before expansion. Existing atoms were preserved.",
             f"Completion orientation: {completion['orientation']}; measured repeat: {completion['repeat_units']} chain units."]
    lines.extend(f"Protofilament #{a['protofilament']}: copied {a['source_chain']} to {a['added_chain']} at the {a['side']} end."
                 for a in completion['additions'])
    lines.extend(completion.get('placement_notes', []))
    return lines


def _expand_unequal_stacks(editor, sandwiches, layers_to_add, use_auto, manual_twist,
                          manual_rise, water_z_limit, use_alt, use_computed_axis,
                          label_generator, repeat_units, random_seed):
    import numpy as np
    ca = editor.ca_chains()
    detection = detect_layers({c: ca[c] for s in sandwiches for c in s})
    if (not terminal_completion_supported(detection) or
            {tuple(s) for s in detection['sandwiches']} != {tuple(s) for s in sandwiches}):
        raise StructureEditError('Automatic completion requires contiguous, ordered stacks with supported terminal deficits, '
                                 'determined strand orientation and one physical layer per chain. Internal gaps and ambiguous stacks are not repaired.')
    minimum, depth = min(map(len, sandwiches)), max(map(len, sandwiches))
    common = [s[:minimum] for s in sandwiches]
    period = _expansion_repeat_pattern(ca, common)['period']
    if minimum <= period:
        raise StructureEditError('Terminal completion needs an observed same-phase repeat in every protofilament.')
    # Ask the established expander for both terminal candidates. Its source is
    # still the whole assembly: the longer stacks' extra atoms are never cut.
    probe = _edit_clone(editor)
    repair_result = expand_structure_layers(probe, common, 2 * (depth - minimum), True, 0., 4.8,
        water_z_limit, label_generator=label_generator, repeat_units=period)
    geometry = probe.report['geometry']
    rotation, translation = np.array(geometry['cycle_matrix']), np.array(geometry['cycle_vector'])
    axis = np.array(geometry['axis'])
    # Validate every observed pair, including the fifth chains omitted from
    # the common-length fitting view. A pooled core may not hide a bad stack.
    errors = []
    for stack in sandwiches:
        for i in range(len(stack) - period):
            p, q = _repeat_coordinate_pairs(ca[stack[i]], ca[stack[i + period]])
            fit = _fit_detection_core(p, q)
            if fit is None:
                raise StructureEditError('No stable repeat supports terminal completion.')
            residual = np.linalg.norm(p @ rotation.T + translation - q, axis=1)[fit['mask']]
            errors.append(float(np.sqrt(np.mean(residual ** 2))))
    if max(errors) > 1.5:
        raise StructureEditError('The protofilaments do not share a consistent measured repeat; terminal completion was not applied.')
    centers = {c: np.mean([r['coord'] for r in ca[c]], axis=0) for c in ca}
    midplane = float(np.median([np.mean([centers[c] @ axis for c in s])
                               for s in sandwiches if len(s) == depth]))
    candidates = defaultdict(dict)
    for choice, group in zip(probe.report['geometry']['template_choices'], probe.groups[1:]):
        dest = choice['destination_unit']
        for pf, (donor, copied) in enumerate(zip(choice['source_chains'], choice['destination_chains'])):
            if len(sandwiches[pf]) == depth:
                continue
            candidates[pf][dest] = (dest, donor, copied, group)
    choices = {}
    placement_notes = []
    for pf, options in candidates.items():
        size = len(sandwiches[pf])
        missing = depth - size
        alternatives = []
        for lower in range(missing + 1):
            destinations = list(range(-lower, 0)) + list(range(size, size + missing - lower))
            if not all(i in options for i in destinations):
                continue
            chosen = [options[i] for i in destinations]
            extra_centers = [g['rotation'] @ centers[donor] + g['translation']
                             for _, donor, _, g in chosen]
            midpoint = (sum(centers[c] @ axis for c in sandwiches[pf]) +
                        sum(center @ axis for center in extra_centers)) / depth
            alternatives.append((abs(float(midpoint) - midplane), tuple(destinations), chosen))
        if not alternatives:
            raise StructureEditError('No contiguous terminal copies can complete the short stack.')
        alternatives.sort(key=lambda item: item[:2])
        selected = alternatives[0]
        if minimum < 3 or depth - minimum > 1:
            # Staggered protofilaments can have equally plausible terminations.
            # Prefer balanced ends among equivalent midplane offsets; disclose
            # that boundary choice instead of implying a unique reconstruction.
            close = [item for item in alternatives if item[0] - selected[0] < .25]
            if len(close) > 1:
                selected = min(close, key=lambda item: (abs(sum(i < 0 for i in item[1]) -
                                                            sum(i >= 0 for i in item[1])), item[1]))
                placement_notes.append(f'Protofilament #{pf + 1}: several terminal placements fit the measured repeat; used balanced ends with a consistent lower-end tie break. These are modeled additions, not uniquely determined missing coordinates.')
        choices[pf] = selected[2]
    attached = editor.covalent_residue_owners()
    copies = [dict(chains={c: c for c in editor.chains})]
    completed = [list(s) for s in sandwiches]
    instances = {(pf, i): c for pf, s in enumerate(sandwiches) for i, c in enumerate(s)}
    additions = []
    for pf, chosen in sorted(choices.items()):
        completed[pf] = ([copied for dest, _, copied, _ in chosen if dest < 0] + sandwiches[pf] +
                         [copied for dest, _, copied, _ in chosen if dest >= 0])
        for dest, donor, copied, group in chosen:
            rows = [r for r in editor.atoms if
                    (r['auth_asym_id'] == donor and r['label_asym_id'] in editor.protein_labels)
                    or attached.get(_residue_key(r)) == donor]
            mapping = {c: group['chains'][c] for c in dict.fromkeys(r['auth_asym_id'] for r in rows)}
            copies.append(dict(chains=mapping, atom_ids={r['id'] for r in rows},
                               matrix=group['rotation'], vector=group['translation']))
            instances[pf, dest] = copied
            additions.append(dict(protofilament=pf + 1, side='lower' if dest < 0 else 'upper',
                                  source_chain=donor, added_chain=copied))
    # Reuse only companion candidates on the chosen terminal side of a short
    # stack, and only if completing that stack adds a whole occupancy period.
    owners = editor.associated_residues({c for s in sandwiches for c in s}, water_z_limit, axis)
    locations = {c: pf for pf, s in enumerate(sandwiches) for c in s}
    ligand_groups = probe.groups[1 + len(probe.report['geometry']['template_choices']):]
    for item, group in zip(probe.report['ligand_copies'], ligand_groups):
        owner = owners.get(tuple(item['residue']))
        pf = locations.get(owner)
        ligand_period = item['period_units']
        if (pf not in choices or item['destination_unit'] not in {c[0] for c in choices[pf]} or
                depth // ligand_period == minimum // ligand_period):
            continue
        copies.append(dict(chains=group['chains'], atom_ids=set(group['ids']),
                           matrix=group['rotation'], vector=group['translation']))
    repaired = _edit_clone(editor)
    repaired.apply_copies(copies)
    repaired.propagate_layer_bonds(sandwiches, instances, period)
    repaired.propagate_layer_sheets(sandwiches, instances, period)
    buffer = StructureBuffer('terminal-completion.cif')
    repaired.write(buffer, format='cif')
    staged = StructureEditor(buffer)
    checked = detect_layers(staged.ca_chains())
    expected = {frozenset(s) for s in completed}
    original_orientation = {frozenset(t['chains']): t['orientation'] for t in detection['protofilaments']}
    expected_orientation = {frozenset(new): original_orientation[frozenset(old)]
                            for old, new in zip(sandwiches, completed)}
    if (not checked['complete'] or checked['detected_units'] != depth or
            {frozenset(s) for s in checked['sandwiches']} != expected or
            any(t['orientation'] != expected_orientation.get(frozenset(t['chains']))
                for t in checked['protofilaments'])):
        raise StructureEditError('Modeled terminal copies did not produce the expected complete stacks; no edit applied.')
    # Reject overlap with another scaffold, not just errors in our own track.
    repaired_ca = staged.ca_chains()
    if not _expansion_repeat_pattern(repaired_ca, completed)['compatible'](period):
        raise StructureEditError('Terminal completion changed the source orientation motif; no edit applied.')
    new_ids = {a['added_chain'] for a in additions}
    for cid in new_ids:
        p = np.array([r['coord'] for r in repaired_ca[cid]])
        q = np.array([r['coord'] for c, rows in repaired_ca.items() if c != cid for r in rows])
        if np.any(np.sum((p[:, None, :] - q[None, :, :]) ** 2, axis=2) < 2.5 ** 2):
            raise StructureEditError('A modeled terminal chain overlaps the protein scaffold; no edit applied.')
    # Continue from the completed assembly, preserving the requested expansion
    # options. Completion itself always uses the measured source repeat.
    if layers_to_add:
        result = expand_structure_layers(staged, completed, layers_to_add, use_auto, manual_twist,
            manual_rise, water_z_limit, use_alt, use_computed_axis, label_generator,
            repeat_units, random_seed)
    else:
        staged.apply_copies([dict(chains={c: c for c in staged.chains})])
        result = repair_result
        staged.report['geometry'] = deepcopy(geometry)
        staged.report['geometry']['template_choices'] = []
    # Compose both stages back onto original atoms. Donor/style mappings and
    # metadata must refer to original chains, never to temporary repair chains.
    composed = []
    for final_group in staged.groups:
        for first_group in repaired.groups:
            ids = {old: middle for old, middle in first_group['ids'].items()
                   if middle in final_group['ids']}
            if not ids:
                continue
            source_chains = {r['auth_asym_id'] for r in first_group['rows'] if r['id'] in ids}
            mapping = {c: final_group['chains'][first_group['chains'][c]] for c in source_chains}
            composed.append(dict(chains=mapping, atom_ids=set(ids),
                matrix=final_group['rotation'] @ first_group['rotation'],
                vector=final_group['rotation'] @ first_group['translation'] + final_group['translation']))
    final = _edit_clone(editor)
    final.report['operation'] = 'complete terminal layers and expand'
    final.report['geometry'] = deepcopy(staged.report['geometry'])
    for field in ('automatic_fit', 'ligand_repeats', 'ligand_copies'):
        if field in staged.report:
            final.report[field] = deepcopy(staged.report[field])
    final.report['warnings'] = list(dict.fromkeys(final.report['warnings'] + repaired.report['warnings'] + staged.report['warnings']))
    final.report['terminal_completion'] = dict(original_lengths=[len(s) for s in sandwiches],
        completed_units=depth, added_chain_units=len(additions), additions=additions,
        repeat_units=period, orientation=detection['orientation'],
        method='measured same-phase repeat; terminal side minimizes offset from the longer stacks midplane',
        modeled=True, pair_core_rmsds=errors, geometry=deepcopy(geometry))
    if placement_notes:
        final.report['terminal_completion']['placement_notes'] = placement_notes
    final.apply_copies(composed)
    _check_new_ca_overlaps(final, reject=not use_auto)
    # Align source indices to their positions in the completed five-layer view.
    padded = [[None] * sum(c[0] < 0 for c in choices.get(pf, [])) + list(s)
              for pf, s in enumerate(sandwiches)]
    final_instances = {(pf, i): c for pf, s in enumerate(completed) for i, c in enumerate(s)}
    for choice in staged.report['geometry']['template_choices']:
        for pf, c in enumerate(choice['destination_chains']):
            final_instances[pf, choice['destination_unit']] = c
    final.propagate_layer_bonds(padded, final_instances, period)
    final.propagate_layer_sheets(padded, final_instances, period)
    source = {_atom_key(r): r for r in editor.atoms}
    retained = {_atom_key(r): r for r in final.output_atoms if _atom_key(r) in source}
    if retained.keys() != source.keys() or any(retained[k][f] != row[f]
            for k, row in source.items() for f in ('Cartn_x', 'Cartn_y', 'Cartn_z')):
        raise StructureEditError('Terminal completion changed an existing atom; no edit applied.')
    final.report['source_preservation'] = dict(atoms=len(source), coordinates='unchanged')
    editor.__dict__.update(final.__dict__)
    return result


def _rotation_about_axis(p, q, axis):
    """Least-squares proper rotation with a prescribed unit axis."""
    _, np = _structure_dependencies()
    pp, qq = p - p.mean(0), q - q.mean(0)
    pp -= (pp @ axis)[:, None] * axis
    qq -= (qq @ axis)[:, None] * axis
    angle = math.atan2(float(np.sum(np.cross(pp, qq) @ axis)), float(np.sum(pp * qq)))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _check_new_ca_overlaps(editor, *, reject=False):
    """Diagnose near-coincident C-alpha atoms across newly placed copies.

    This small-distance check is a geometry sanity check, not an energy model or
    a substitute for all-atom refinement/validation. Alternate conformers of the
    same residue and independent ensemble models are never compared.
    """
    threshold = MIN_NEW_CA_SEPARATION
    original = {_atom_key(row) for row in editor.atoms}
    protein_labels = {new for group in editor.groups for old, new in group['labels'].items()
                      if old in editor.protein_labels}
    copy_groups = {atom_id: index for index, group in enumerate(editor.groups)
                   for atom_id in group['ids'].values()}
    buckets = defaultdict(list)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    examined = 0
    clashes = []
    clash_count = 0
    for row in editor.output_atoms:
        if row['label_atom_id'] != 'CA' or row['label_asym_id'] not in protein_labels:
            continue
        xyz = tuple(float(row[f]) for f in ('Cartn_x', 'Cartn_y', 'Cartn_z'))
        cell = tuple(math.floor(x / threshold) for x in xyz)
        model = str(row['pdbx_PDB_model_num'])
        is_new = _atom_key(row) not in original
        residue = _residue_key(row)
        copy_group = copy_groups[row['id']]
        for dx, dy, dz in offsets:
            for other, other_new, other_residue, other_group in buckets.get((model, cell[0]+dx, cell[1]+dy, cell[2]+dz), ()):
                if ((is_new or other_new) and copy_group != other_group
                        and residue[0] != other_residue[0]):
                    distance2 = sum((a-b)**2 for a, b in zip(xyz, other))
                    if distance2 < threshold**2:
                        clash_count += 1
                        if len(clashes) < 10:
                            clashes.append(dict(residue=list(residue), other_residue=list(other_residue),
                                                distance_angstrom=math.sqrt(distance2)))
        buckets[(model,) + cell].append((xyz, is_new, residue, copy_group))
        examined += 1
    editor.report['overlap_check'] = dict(ca_atoms=examined, minimum_new_separation_angstrom=threshold,
                                        close_pairs=clash_count, examples=clashes,
                                        scope='C-alpha contacts across new copies only; not all-atom clash validation')
    if clash_count:
        message = (f'Expansion introduces C-alpha atoms less than {threshold:g} Å apart '
                   f'across placed copies ({clash_count} pairs). Review repeat geometry and source contacts.')
        if reject:
            raise StructureEditError(message + ' Manual expansion was not applied.')
        # Flexible experimental templates may have poor terminal contacts. Report
        # these explicitly without removing the existing automatic-edit workflow.
        editor.report['warnings'].append(message)


def _fit_repeat_transform(ca, stacks, period, axis, centers, *, computed_axis, randomize) -> RepeatTransform:
    """Fit a cycle to stable C-alpha cores and report support for every pair.

    ``period`` counts chain units. Returned twist/rise are per chain unit, not
    necessarily per physical layer. The input axis is an observed repeat direction;
    a resolved unconstrained rotation may supply its own screw axis instead.
    """
    _, np = _structure_dependencies()
    sandwiches, step, depth = stacks, period, len(stacks[0])
    use_computed_axis = computed_axis
    axis = axis.copy()
    observed_axis = axis.copy()
    if depth <= step:
        raise StructureEditError(f'Automatic fitting for a {step}-unit repeat needs at least {step + 1} source chain units. '
                                 'Load another repeat or enter manual twist/rise.')
    source, target, pair_ranges, repeat_directions = [], [], [], []
    for s in sandwiches:
        for i in range(depth - step):
            pair_p, pair_q = _repeat_coordinate_pairs(ca[s[i]], ca[s[i + step]])
            pair_ranges.append((len(source), len(source) + len(pair_p)))
            source.extend(pair_p)
            target.extend(pair_q)
            if use_computed_axis and (step > 1 or randomize) and len(pair_p) >= 6:
                fit = _fit_detection_core(pair_p, pair_q)
                if fit is None or fit['rmsd'] > 1.5:
                    raise StructureEditError('No stable core between equivalent repeat positions.')
                repeat_directions.append((pair_q - pair_p)[fit['mask']].mean(0))
    p, q = np.asarray(source), np.asarray(target)
    full_p, full_q = p, q
    total_fit_atoms = len(p)
    # Fit the stable core; copy complete chains, including their flexible regions.
    if use_computed_axis and len(p) >= 6:
        core_fit = _fit_detection_core(p, q)
        if core_fit is None or core_fit['rmsd'] > 1.0:
            raise StructureEditError('No consistent stable core for automatic expansion.')
        p, q = p[core_fit['mask']], q[core_fit['mask']]
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    if len(p) < 3 or np.linalg.matrix_rank(p - pc) < 2:
        raise StructureEditError("At least three noncollinear matched C-alpha atoms are needed.")
    axis_constrained = False
    axis_constraint_reason = None
    rigid_cycle_max_error = None
    if use_computed_axis:
        # A rigid cycle spanning several chain units is still one proper fit.
        # Mean atom displacement is not its screw axis: off-axis atoms also
        # move tangentially. Flexible cycles retain their directional model;
        # unresolved near-zero rotations use the observed repeat direction.
        u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
        fix = np.eye(3)
        fix[2, 2] = np.linalg.det(vt.T @ u.T)
        rotation = vt.T @ fix @ u.T
        free_translation = qc - rotation @ pc
        rigid_cycle_max_error = float(np.linalg.norm(full_p @ rotation.T + free_translation - full_q, axis=1).max())
        cycle_angle = math.degrees(math.acos(float(np.clip((np.trace(rotation)-1)/2, -1, 1))))
        if (step > 1 or randomize) and rigid_cycle_max_error > RIGID_REPEAT_COORDINATE_TOLERANCE:
            # Flexible multi-phase templates retain the established directional
            # model. An exact rigid cycle does not need that extra assumption:
            # require EVERY matched CA, not just a trimmed core, to agree within
            # coordinate-rounding tolerance before using its unconstrained axis.
            if repeat_directions:
                axis = np.mean(repeat_directions, axis=0)
                norm = np.linalg.norm(axis)
                if norm < 1e-8:
                    raise StructureEditError('Same-phase repeat directions cancel; check the ordered stacks.')
                axis /= norm
            rotation = _rotation_about_axis(p, q, axis)
            axis_constrained = True
            axis_constraint_reason = 'periodic or randomized repeat'
        elif cycle_angle < MIN_RESOLVED_ROTATION_DEGREES:
            # Near identity, a rotation eigenvector is mostly coordinate noise.
            # Keep the measured repeat axis and refit around that direction.
            rotation = _rotation_about_axis(p, q, axis)
            axis_constrained = True
            axis_constraint_reason = 'near-zero cycle rotation'
        else:
            vals, vecs = np.linalg.eig(rotation)
            axis = np.real(vecs[:, np.argmin(abs(vals - 1))])
            axis /= np.linalg.norm(axis)
            if np.dot(axis, centers[-1] - centers[0]) < 0:
                axis = -axis
    else:
        pp, qq = p - pc, q - qc
        angle = math.atan2(np.sum(pp[:, 0] * qq[:, 1] - pp[:, 1] * qq[:, 0]),
                           np.sum(pp[:, 0] * qq[:, 0] + pp[:, 1] * qq[:, 1]))
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    translation = qc - rotation @ pc
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]) / 2
    twist = math.degrees(math.atan2(float(np.dot(skew, axis)), float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))) / step
    rise = float(np.dot(translation, axis)) / step
    rmsd = float(np.sqrt(np.mean(np.sum((p @ rotation.T + translation - q) ** 2, axis=1))))
    # A pooled trimmed fit must not discard an entire protein/phase.
    errors = np.linalg.norm(full_p @ rotation.T + translation - full_q, axis=1)
    phase_errors = []
    for first, last in pair_ranges:
        pair_fit = _fit_detection_core(full_p[first:last], full_q[first:last])
        mask = pair_fit['mask'] if pair_fit is not None else np.arange(last - first)
        core_error = float(np.sqrt(np.mean(errors[first:last][mask] ** 2)))
        phase_errors.append(core_error)
        if core_error > MAX_REPEAT_CORE_RMSD and (step > 1 or randomize):
            raise StructureEditError('The source units do not share a consistent repeat transform in every phase; choose a different period or manual parameters.')
    diagnostics = {'method': 'stable-core proper rigid fit' if use_computed_axis else 'Z-axis fit',
                                    'matched_ca_atoms': total_fit_atoms, 'fitted_ca_atoms': len(p),
                                    'core_rmsd': rmsd, 'period_units': step,
                                    'axis_constrained_to_repeats': axis_constrained,
                                    'axis_constraint_reason': axis_constraint_reason,
                                    'minimum_resolved_cycle_angle_degrees': MIN_RESOLVED_ROTATION_DEGREES}
    diagnostics['unconstrained_cycle_max_error_angstrom'] = rigid_cycle_max_error
    diagnostics['rigid_cycle_coordinate_tolerance_angstrom'] = RIGID_REPEAT_COORDINATE_TOLERANCE
    diagnostics['pair_core_rmsds'] = phase_errors

    diagnostics['axis_repeat_alignment'] = float(abs(np.dot(axis, observed_axis)))
    diagnostics['warnings'] = []
    if axis_constraint_reason == 'near-zero cycle rotation':
        diagnostics['warnings'].append(
            'Near-zero rotation: the axis is constrained to the observed repeat direction. '
            'The tiny fitted rotation does not independently determine a screw axis.')
    if max(phase_errors, default=0) > MAX_REPEAT_CORE_RMSD:
        diagnostics['warnings'].append(
            f'At least one source pair has stable-core RMSD above {MAX_REPEAT_CORE_RMSD:g} Å '
            'under the shared transform. Flexible boundaries may not follow an ideal repeat; '
            'review the per-pair RMSDs and extrapolated contacts.')
    if computed_axis and diagnostics['axis_repeat_alignment'] < .5:
        diagnostics['warnings'].append(
            'The fitted screw axis differs by more than 60 degrees from the observed repeat direction. '
            'Axis and rise may be unreliable despite a small coordinate RMSD; review the source stacks.')
    return RepeatTransform(rotation, translation, axis, twist, rise, rmsd, diagnostics)


def expand_structure_layers(editor, sandwiches, layers_to_add, use_auto, manual_twist,
                            manual_rise, water_z_limit=4.0, use_alt=False,
                            use_computed_axis=True, label_generator=None,
                            repeat_units=None, random_seed=None):
    # Keep every input atom fixed; use one shared transform only for added copies.
    # repeat_units selects the cycle length; zero or negative values sample compatible templates.
    # Twist and rise are per chain unit, even for chains spanning several layers.
    _, np = _structure_dependencies()
    unequal = bool(sandwiches) and len({len(s) for s in sandwiches}) > 1
    if not isinstance(layers_to_add, int) or layers_to_add < 0 or (layers_to_add == 0 and not unequal):
        raise StructureEditError("Number of added layers must be a positive integer.")
    if not sandwiches or any(not s for s in sandwiches):
        raise StructureEditError("Expansion requires nonempty, ordered protofilaments.")
    core = [c for s in sandwiches for c in s]
    if len(set(core)) != len(core):
        raise StructureEditError("A chain occurs in more than one protofilament/layer.")
    ca = editor.ca_chains()
    if set(core) - set(ca):
        raise StructureEditError("Some selected chains have no polymer C-alpha atoms.")
    if unequal:
        return _expand_unequal_stacks(editor, sandwiches, layers_to_add, use_auto, manual_twist,
            manual_rise, water_z_limit, use_alt, use_computed_axis, label_generator, repeat_units, random_seed)
    depth = len(sandwiches[0])
    pattern = _expansion_repeat_pattern(ca, sandwiches)
    if repeat_units is None:
        repeat_units = 2 if use_alt else pattern['period']
    if isinstance(repeat_units, bool) or not isinstance(repeat_units, int):
        raise StructureEditError('Alternating period must be a whole number; zero or any negative number selects random.')
    randomize = repeat_units <= 0
    if randomize:
        repeat_units = -1  # Canonical audit value for every random-mode alias.
    step = pattern['period'] if randomize else repeat_units
    if step > depth:
        raise StructureEditError(f'A repeat of {step} chain units requires at least {step} source units; only {depth} are available.')
    if not pattern['compatible'](step):
        raise StructureEditError('The repeat period would exchange different protein identities or opposite backbone orientations. '
                                 f'Use a compatible cycle (suggested: {pattern["period"]} chain units).')
    rng = None
    if randomize:
        if random_seed is None:
            random_seed = int(np.random.SeedSequence().entropy)
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise StructureEditError('Random seed must be a nonnegative integer.')
        rng = np.random.default_rng(random_seed)
    centers = np.array([np.mean([r["coord"] for s in sandwiches for r in ca[s[i]]], axis=0) for i in range(depth)])
    center = centers.mean(axis=0)
    axis = np.array([0., 0., 1.])
    axis_source = 'global Z'
    axis_eigenvalues = None
    if use_computed_axis and depth == 1:
        axis, axis_eigenvalues = _single_unit_axis([r['coord'] for c in core for r in ca[c]])
        axis_source = 'single-unit C-alpha PCA plane normal (shape estimate)'
    if use_computed_axis and depth >= 2:
        axis_source = 'source repeats'
        # Same-phase displacements avoid the transverse stagger between A/B.
        repeated = centers[step:] - centers[:-step]
        axis_points = repeated if step > 1 and len(repeated) else centers - center
        _, singular, vt = np.linalg.svd(axis_points)
        if singular[0] < 1e-8:
            raise StructureEditError("Layer centers coincide; helical axis is indeterminate.")
        axis = vt[0]
        if not len(repeated) and pattern['axis'] is not None:
            axis = pattern['axis']
        if np.dot(axis, centers[-1] - centers[0]) < 0:
            axis = -axis
    rmsd = None
    if use_auto:
        rotation, translation, axis, twist, rise, rmsd, fit_report = _fit_repeat_transform(
            ca, sandwiches, step, axis, centers, computed_axis=use_computed_axis, randomize=randomize)
        editor.report['automatic_fit'] = fit_report
        editor.report['warnings'].extend(fit_report['warnings'])
        if use_computed_axis:
            axis_source = ('source repeats; ' + fit_report['axis_constraint_reason']
                           if fit_report['axis_constrained_to_repeats'] else 'stable-core screw axis')
    else:
        twist, rise = float(manual_twist), float(manual_rise)
        if not math.isfinite(twist) or not math.isfinite(rise) or abs(rise) < 1e-8:
            raise StructureEditError("Twist/rise must be finite, with nonzero rise.")
        angle = math.radians(twist * step)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
        translation = center - rotation @ center + rise * step * axis
    if abs(rise) < 1e-8:
        raise StructureEditError("Estimated helical rise is zero; refusing overlapping copies.")
    attached_residues = editor.covalent_residue_owners()
    ligand_sites = editor.associated_residue_repeats(sandwiches, axis, rise, water_z_limit)
    for site in ligand_sites:
        site['donors'] = [r for r in site['residues'] if not any(k in attached_residues for k in r['keys'])]
        site['attached_copies'] = 0
        site['occupied'] = {r['layer'] for r in site['residues']}
    attached_sites = {k: (site, r) for site in ligand_sites for r in site['residues']
                      for k in r['keys'] if k in attached_residues}
    used = set(editor.chains)
    next_index = 0
    def allocate():
        nonlocal next_index
        while True:
            label = (label_generator or structure_chain_id)(next_index)
            next_index += 1
            if label not in used:
                used.add(label)
                return label
    # The identity copy retains all existing atoms; only new chains are transformed.
    copies = [{"chains": {c: c for c in editor.chains}}]
    template_choices = []
    instances = {(pf, layer): chain for pf, strand in enumerate(sandwiches) for layer, chain in enumerate(strand)}
    top = layers_to_add // 2
    bottom = layers_to_add - top
    alignment_cache = {}
    compatibility_cache = {}
    source_at = {i: i for i in range(depth)}
    # Sample outwards, checking compatibility with each placed same-phase neighbor.
    lower = list(range(-1, -bottom - 1, -1)) if randomize else list(range(-bottom, 0))
    for dest in lower + list(range(depth, depth + top)):
        anchor = dest % step if dest < 0 else depth - step + (dest - depth) % step
        alignment = np.eye(4)
        if randomize:
            neighbor = source_at[dest + step if dest < 0 else dest - step]
            eligible = []
            for candidate in range(dest % step, depth, step):
                key = tuple(sorted((candidate, neighbor)))
                if key not in compatibility_cache:
                    compatible = True
                    for stack in sandwiches:
                        if candidate == neighbor:
                            continue
                        p, q = _repeat_coordinate_pairs(ca[stack[candidate]], ca[stack[neighbor]])
                        fit = _fit_detection_core(p, q) if len(p) >= 3 else None
                        if fit is None or fit['rmsd'] > 1.0:
                            compatible = False
                            break
                    compatibility_cache[key] = compatible
                if compatibility_cache[key]:
                    eligible.append(candidate)
            ref = eligible[int(rng.integers(len(eligible)))]
            # Align sampled copies to the boundary core to avoid accumulating donor-position errors.
            if ref != anchor:
                key = ref, anchor
                if key not in alignment_cache:
                    donor, reference = [], []
                    for stack in sandwiches:
                        p, q = _repeat_coordinate_pairs(ca[stack[ref]], ca[stack[anchor]])
                        donor.extend(p)
                        reference.extend(q)
                    p, q = np.array(donor), np.array(reference)
                    fit = _fit_detection_core(p, q) if len(p) >= 6 else None
                    if fit is None or fit['rmsd'] > 1.5:
                        raise StructureEditError('A sampled unit has no consistent stable core with its repeat position.')
                    p, q = p[fit['mask']], q[fit['mask']]
                    pc, qc = p.mean(0), q.mean(0)
                    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
                    fix = np.eye(3)
                    fix[2, 2] = np.linalg.det(vt.T @ u.T)
                    r = vt.T @ fix @ u.T
                    transform = np.eye(4)
                    transform[:3, :3], transform[:3, 3] = r, qc - r @ pc
                    alignment_cache[key] = transform
                alignment = alignment_cache[key]
        else:
            ref = anchor
        source_at[dest] = ref
        power = (dest - anchor) // step
        homogeneous = np.eye(4)
        homogeneous[:3, :3], homogeneous[:3, 3] = rotation, translation
        transformed = np.linalg.matrix_power(homogeneous, power) @ alignment
        source_chains = {s[ref] for s in sandwiches}
        selected = [r for r in editor.atoms if (_null_id(r["auth_asym_id"]) in source_chains and
                    r['label_asym_id'] in editor.protein_labels) or attached_residues.get(_residue_key(r)) in source_chains]
        chain_order = list(dict.fromkeys(_null_id(r["auth_asym_id"]) for r in selected))
        mapping = {c: allocate() for c in chain_order}
        for pf, strand in enumerate(sandwiches):
            instances[pf, dest] = mapping[strand[ref]]
        template_choices.append(dict(destination_unit=dest, source_unit=ref, anchor_unit=anchor,
                                     eligible_templates=len(eligible) if randomize else 1,
                                     source_chains=[s[ref] for s in sandwiches],
                                     destination_chains=[instances[pf, dest] for pf in range(len(sandwiches))]))
        copies.append({"chains": mapping, "atom_ids": {r["id"] for r in selected},
                       "matrix": transformed[:3, :3], "vector": transformed[:3, 3]})
        copied_attachments = set()
        for row in selected:
            key = _residue_key(row)
            if key in attached_sites:
                site, residue = attached_sites[key]
                if residue['key'] not in copied_attachments:
                    copied_attachments.add(residue['key'])
                    site['attached_copies'] += 1
                    site['occupied'].add(residue['layer'] + dest - ref)
    ligand_copies = []
    homogeneous = np.eye(4)
    homogeneous[:3, :3], homogeneous[:3, 3] = rotation, translation
    # Include retained ligands and protein-attached copies in duplicate checks.
    placed = defaultdict(list)
    for site in ligand_sites:
        for residue in site['residues']:
            placed[site['identity']].append(residue['xyz'])
            if any(k in attached_residues for k in residue['keys']):
                atom_ids = {r['id'] for r in editor.atoms if _residue_key(r) in residue['keys']}
                for spec in copies[1:]:
                    if atom_ids.issubset(spec['atom_ids']):
                        placed[site['identity']].append(residue['xyz'] @ spec['matrix'].T + spec['vector'])
    for site in ligand_sites:
        if not site['donors']:
            continue
        period = site['period']
        count = max(0, (depth + layers_to_add) // period - depth // period - site['attached_copies'])
        candidates = []
        occupied = site['occupied']
        bounds = range(-bottom, depth + top)
        if site['terminal_track']:
            first = min(r['layer'] for r in site['residues'])
            last = max(r['layer'] for r in site['residues'])
            bounds = list(range(first - bottom, first)) + list(range(last + 1, last + top + 1))
        for dest in bounds:
            if dest in occupied:
                continue
            donors = [r for r in site['donors'] if (dest - r['layer']) % period == 0 and
                      (dest - r['layer']) % step == 0]
            if donors:
                donor = min(donors, key=lambda r: abs(dest - r['layer']))
                candidates.append((dest, donor))
        midpoint = (depth + top - bottom - 1) / 2
        candidates.sort(key=lambda item: (abs(item[0] - midpoint), item[0]))
        added = 0
        overlaps = 0
        for dest, donor in candidates:
            if added >= count:
                break
            selected = [r for r in editor.atoms if _residue_key(r) in donor['keys']]
            # matrix_power(H, 1) can return H itself. Keep each copy independent
            # so a per-copy change can never corrupt the shared screw transform.
            transformed = np.linalg.matrix_power(homogeneous, (dest - donor['layer']) // step).copy()
            # A ligand's entire pose must follow its protein repeat. Freezing
            # its orientation while rotating its center changes pocket contacts,
            # even when the deposited ligand copies happen to be parallel.
            xyz = donor['xyz'] @ transformed[:3, :3].T + transformed[:3, 3]
            if len(xyz) >= 6 and any(p.shape == xyz.shape and
                    np.linalg.norm(xyz.mean(0) - p.mean(0)) < min(2., .4 * abs(rise)) and
                    np.sqrt(np.mean(np.sum((xyz - p) ** 2, axis=1))) < min(2., .4 * abs(rise))
                    for p in placed[site['identity']]):
                editor.report['warnings'].append('Skipped an overlapping ligand copy of ' + site['identity'][0])
                overlaps += 1
                continue
            placed[site['identity']].append(xyz)
            mapping = {chain: allocate() for chain in dict.fromkeys(r['auth_asym_id'] for r in selected)}
            copies.append(dict(chains=mapping, atom_ids={r['id'] for r in selected},
                               matrix=transformed[:3, :3], vector=transformed[:3, 3]))
            ligand_copies.append(dict(residue=list(donor['key']), source_unit=donor['layer'],
                                      destination_unit=dest, period_units=period,
                                      orientation='protein screw'))
            added += 1
        if overlaps and added < count:
            raise StructureEditError('Ligand repeat sites propose overlapping copies of ' + site['identity'][0] +
                                     '; cannot complete expansion without changing the inferred occupancy. No edit applied.')
    editor.report['ligand_repeats'] = [dict(component=s['identity'][0], period_units=s['period'],
                                          source_residues=[list(r['key']) for r in s['residues']]) for s in ligand_sites]
    editor.report['ligand_copies'] = ligand_copies
    editor.report["operation"] = "expand layers"
    editor.report["geometry"] = {"twist_degrees": twist, "rise_angstrom": rise, "axis": axis.tolist(),
                                 "axis_source": axis_source,
                                 "axis_pca_variances_angstrom2": axis_eigenvalues.tolist() if axis_eigenvalues is not None else None,
                                 "fit_rmsd_angstrom": rmsd, "alternating": step > 1,
                                 "repeat_units": repeat_units, "transform_period_units": step,
                                 "random_seed": random_seed if randomize else None,
                                 "template_choices": template_choices,
                                 "cycle_matrix": rotation.tolist(), "cycle_vector": translation.tolist(),
                                 "manual_axis_point": center.tolist() if not use_auto else None}
    editor.apply_copies(copies)
    _check_new_ca_overlaps(editor, reject=not use_auto)
    editor.propagate_layer_bonds(sandwiches, instances, step)
    editor.propagate_layer_sheets(sandwiches, instances, step)
    source_atoms = {_atom_key(row): row for row in editor.atoms}
    retained = {_atom_key(row): row for row in editor.output_atoms if _atom_key(row) in source_atoms}
    if retained.keys() != source_atoms.keys() or any(
            retained[key][field] != row[field]
            for key, row in source_atoms.items() for field in ('Cartn_x', 'Cartn_y', 'Cartn_z')):
        raise StructureEditError('Expansion changed an existing atom; the edit was rejected.')
    editor.report['source_preservation'] = {'atoms': len(source_atoms), 'coordinates': 'unchanged'}
    tilt = math.degrees(math.acos(float(np.clip(abs(axis[2]), -1, 1))))
    return twist, rise, axis, tilt


def protofilament_selection_targets(editor, detection, axial_limit=4.0):
    """Residue-exact targets; author chains can contain both protein and ligand."""
    _, np = _structure_dependencies()
    stacks = detection['sandwiches']
    first = [r for r in editor.atoms if str(r['pdbx_PDB_model_num']) == editor.models[0]]
    targets = []
    for i, stack in enumerate(stacks):
        targets.append(dict(kind='protein', pf=i,
            label=f'Protein protofilament with Chain {stack[len(stack)//2]}',
            residues=frozenset(_residue_key(r) for r in first
                if r['auth_asym_id'] in stack and r['label_asym_id'] in editor.protein_labels)))

    # Reuse the editor's site/occupancy detection, retaining complete RNA/DNA
    # molecules too. Waters and monatomic ions are not ligand protofilaments.
    molecules = [rows for rows in editor.companion_groups()
                 if len({r['label_atom_id'] for r in rows}) > 1
                 and rows[0]['label_comp_id'] not in ('HOH', 'WAT', 'DOD')]
    eligible = {_residue_key(r) for rows in molecules for r in rows}
    rises = [p['unit_rise'] for p in detection['protofilaments'] if p['unit_rise']]
    sites = (editor.associated_residue_repeats(stacks, detection['axis'],
                float(np.median(rises)), axial_limit)
             if rises and detection['axis'] is not None else [])
    assigned = set()
    for site in sites:
        keys = frozenset(k for member in site['residues'] for k in member['keys']) & eligible
        if keys:
            targets.append(dict(kind='ligand', residues=keys, copies=len(site['residues'])))
            assigned.update(keys)
    # Do not silently omit molecules outside the axial association window or
    # in a one-layer fragment where the stacking axis cannot be determined.
    for rows in molecules:
        keys = frozenset(_residue_key(r) for r in rows)
        if not keys & assigned:
            targets.append(dict(kind='ligand', residues=keys, copies=1))
    for number, target in enumerate((t for t in targets if t['kind'] == 'ligand'), 1):
        keys = sorted(target['residues'])
        names = '/'.join(sorted({k[-1] for k in keys}))
        anchor = keys[0]
        target['label'] = (f'Ligand {names} site {number} ({target["copies"]} copies; '
                           f'{anchor[0]}:{anchor[1]}{anchor[2]})')
    return targets


def trim_ligand_target(editor, target):
    keys = target['residues']
    if keys & editor.covalent_residue_owners().keys():
        raise StructureEditError('This ligand site is covalently attached to protein; it cannot be trimmed independently.')
    rows = [r for r in editor.atoms if _residue_key(r) not in keys]
    editor.report['operation'] = 'trim ligand site'
    return editor.apply_copies([dict(chains={r['auth_asym_id']: r['auth_asym_id'] for r in rows},
                                    atom_ids={r['id'] for r in rows})])


def new_source_notices(owner, report):
    """Display each source issue once per explicit load; keep the full audit."""
    seen = getattr(owner, '_shown_source_notices', None)
    if seen is None:
        owner._shown_source_notices = seen = set()
    for issue in report.get('source_issues', []):
        key = (issue['kind'], issue['warning'])
        if key in seen:
            continue
        seen.add(key)
        message = issue['warning']
        if issue['kind'] == 'unreviewed_cif_category':
            message += ' This is an original-source annotation; it can also be carried in a PDB working copy.'
        yield message


from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, 
                             QPushButton, QTextEdit, QMessageBox, QFileDialog, 
                             QVBoxLayout, QHBoxLayout, QSizePolicy, QSlider,
                             QSplitter, QSpinBox, QGroupBox, QFormLayout, 
                             QCheckBox, QTableWidget, QHeaderView, QTableWidgetItem, 
                             QTextBrowser, QDialog, QTabWidget, QComboBox,
                             QTableView, QAbstractItemView)
from PyQt6.QtCore import (Qt, QTimer, QSize, QAbstractTableModel,
                          QModelIndex, QProcess, QPointF)
from PyQt6.QtGui import QColor

# Pyqtgraph caches OpenGL shader program IDs at class level. Top-level
# QOpenGLWidgets must share resources or a shader compiled by the main preview
# is not valid in a separate quick-look window. Qt requires this attribute to
# be set before QApplication is constructed.
if QApplication.instance() is None:
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts,
        True,
    )


# 1. Catch the arguments passed by xcop.py
parser = argparse.ArgumentParser()
parser.add_argument("--chimera-ver", type=str, default="chimera")
renderer_default = os.environ.get("XCOP_RENDERER", "auto").strip().lower()
if renderer_default not in {"auto", "opengl", "software"}:
    renderer_default = "auto"
parser.add_argument(
    "--renderer",
    choices=("auto", "opengl", "software"),
    default=renderer_default,
    help="3D preview renderer (or set XCOP_RENDERER)",
)
args, unknown = parser.parse_known_args()
CHIMERA_VER = args.chimera_ver

# 2. Recreate required global variables so the script doesn't crash
toplevel_windows = []

def show_error_message(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Error")
    msg.setText(message)
    msg.exec()

# ====================== Multi-layer Chain Modifier Class ======================

def open_modifier_layer_viewer(launch=True):
    try:
        import uuid # Required for unique temp filenames
        import numpy as np  # Required for the new detection logic
        import json
        import subprocess
        import pyqtgraph.opengl as gl
        from OpenGL.GL import glDepthFunc, GL_LEQUAL, GL_LESS
        from PyQt6.QtGui import (
            QDoubleValidator, QIntValidator, QMatrix4x4,
            QVector3D, QVector4D, QOpenGLContext, QPainter, QPen, QBrush,
        )
        from PyQt6.QtWidgets import QToolTip
        from scipy.spatial import cKDTree
    except ImportError as e:
        print(f"Error loading preview dependencies: {e}")
        return

    def display_server_looks_like_xquartz():
        """Detect an SSH-forwarded XQuartz display without relying on sys.platform."""
        if not os.environ.get("DISPLAY"):
            return False
        try:
            result = subprocess.run(
                ["xdpyinfo"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return False
        display_info = f"{result.stdout}\n{result.stderr}".lower()
        return any(marker in display_info for marker in (
            "xquartz", "apple-wm", "apple-dri", "apple computer",
        ))

    renderer = args.renderer
    if renderer == "auto":
        renderer = "software" if display_server_looks_like_xquartz() else "opengl"
    use_software_renderer = renderer == "software"
    print(f"Modifier preview renderer: {renderer}")
    
    def create_outline_folder_icon():
        from PyQt6.QtGui import QPixmap, QIcon, QPainter, QPen, QColor, QPolygon
        from PyQt6.QtCore import Qt, QPoint
        
        # Create a 24x24 transparent canvas
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Apply your theme's light green
        pen = QPen(QColor('#98c379'))
        pen.setWidth(2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        # Draw the folder shape (tab on the left)
        poly = QPolygon([
            QPoint(2, 6),   # Top-left tab
            QPoint(9, 6),   # Top-right tab
            QPoint(11, 9),  # Fold down
            QPoint(22, 9),  # Top-right edge
            QPoint(22, 20), # Bottom-right
            QPoint(2, 20)   # Bottom-left
        ])
        painter.drawPolygon(poly)
        
        # Draw the inner line to give it that 'folder' look
        painter.drawLine(2, 9, 11, 9)
        
        painter.end()
        return QIcon(pixmap)


    def context_safe_shader_program(item_class, program_cache):
        """Return a pyqtgraph shader compiled for the current GL share-group."""
        context = QOpenGLContext.currentContext()
        if (
            context is None
            or not hasattr(item_class, "_shaderProgram")
            or not hasattr(item_class, "getShaderProgram")
        ):
            return item_class.getShaderProgram()

        share_group = context.shareGroup()
        cache_key = share_group if share_group is not None else context
        cached_program = program_cache.get(cache_key)
        if cached_program is not None:
            return cached_program

        previous_program = item_class._shaderProgram
        item_class._shaderProgram = None
        try:
            program = item_class.getShaderProgram()
        finally:
            item_class._shaderProgram = previous_program

        program_cache[cache_key] = program
        try:
            cache_key.destroyed.connect(
                lambda _object=None, key=cache_key, cache=program_cache:
                cache.pop(key, None)
            )
        except (AttributeError, TypeError):
            pass
        return program


    class ContextSafeLinePlotItem(gl.GLLinePlotItem):
        _programs_by_share_group = {}

        @staticmethod
        def getShaderProgram():
            return context_safe_shader_program(
                gl.GLLinePlotItem,
                ContextSafeLinePlotItem._programs_by_share_group,
            )


    class ContextSafeScatterPlotItem(gl.GLScatterPlotItem):
        _programs_by_share_group = {}

        @staticmethod
        def getShaderProgram():
            return context_safe_shader_program(
                gl.GLScatterPlotItem,
                ContextSafeScatterPlotItem._programs_by_share_group,
            )


    class OpenGLChainCanvas(gl.GLViewWidget):
        """OpenGL chain trace preview shared by Modifier and Layer Viewer."""

        LABEL_STYLE = (
            "QLabel { color: black; background-color: rgba(255, 255, 255, 230); "
            "border: 1px solid #333333; border-radius: 3px; padding: 2px; "
            "font-size: 11pt; font-weight: bold; }"
        )

        def __init__(
            self,
            parent=None,
            width=8,
            height=5,
            dpi=50,
            highlight_selected=False,
            center_on_labels=False,
            centered_labels=True,
        ):
            super().__init__(parent, rotationMethod='quaternion')
            # Match the preferred and minimum sizes advertised by the former
            # Matplotlib canvas. QSplitter uses these hints for its startup
            # allocation; QOpenGLWidget otherwise reports an invalid (-1, -1)
            # hint and lets the controls pane consume almost all available
            # width.
            self._preferred_size = QSize(
                max(1, round(width * dpi)),
                max(1, round(height * dpi)),
            )
            self.setBackgroundColor('#2C2C2C')
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.updateGeometry()

            self.highlight_selected = highlight_selected
            self.center_on_labels = center_on_labels
            self.centered_labels = centered_labels
            self.global_center = (0.0, 0.0, 0.0)
            self.max_range = 10.0
            self.zoom_level = 50
            self.slider_callback = None
            self._chain_labels = []
            self._label_update_pending = False
            self.setCameraPosition(distance=30.0, elevation=30, azimuth=-60)

        def sizeHint(self):
            return QSize(self._preferred_size)

        def minimumSizeHint(self):
            return QSize(10, 10)

        def _clear_scene(self):
            self.clear()
            for label, _position in self._chain_labels:
                label.hide()
                label.deleteLater()
            self._chain_labels = []

        def wheelEvent(self, event):
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta == 0:
                event.ignore()
                return
            step = 5 if delta > 0 else -5
            self.zoom_level = max(1, min(100, self.zoom_level + step))
            if self.slider_callback:
                self.slider_callback(self.zoom_level)
            self.apply_zoom()
            event.accept()

        def mouseMoveEvent(self, event):
            super().mouseMoveEvent(event)
            self._schedule_label_update()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._schedule_label_update()

        def set_zoom_from_slider(self, value):
            self.zoom_level = value
            self.apply_zoom()

        def apply_zoom(self):
            if self.max_range == 0:
                return
            factor = 2.0 - ((self.zoom_level / 100.0) * 1.9)
            if factor < 0.05:
                factor = 0.05
            radius = self.max_range * factor
            self.setCameraPosition(
                pos=QVector3D(*self.global_center),
                distance=max(radius * 2.25, 0.05),
            )
            self._schedule_label_update()

        def _schedule_label_update(self):
            if self._label_update_pending:
                return
            self._label_update_pending = True
            QTimer.singleShot(0, self._update_label_positions)

        def _screen_position(self, position):
            width, height = self.width(), self.height()
            if width <= 0 or height <= 0:
                return None
            vector = QVector4D(*position, 1.0)
            region = (0, 0, width, height)
            viewport = (0, 0, width, height)
            clip = (
                self.projectionMatrix(region, viewport)
                * self.viewMatrix()
                * vector
            )
            if clip.w() <= 0:
                return None
            ndc_x = clip.x() / clip.w()
            ndc_y = clip.y() / clip.w()
            if not (-1.05 <= ndc_x <= 1.05 and -1.05 <= ndc_y <= 1.05):
                return None
            return (
                (ndc_x + 1.0) * width / 2.0,
                (1.0 - ndc_y) * height / 2.0,
            )

        def _update_label_positions(self):
            self._label_update_pending = False
            for label, position in self._chain_labels:
                screen_position = self._screen_position(position)
                if screen_position is None:
                    label.hide()
                    continue
                if self.centered_labels:
                    x = int(screen_position[0] - label.width() / 2)
                    y = int(screen_position[1] - label.height() / 2)
                else:
                    # Matplotlib's default text anchor is left/baseline; keep
                    # that placement in Layer Viewer.
                    x = int(screen_position[0])
                    y = int(screen_position[1] - label.height() + 3)
                label.move(x, y)
                label.show()
                label.raise_()

        def _add_label(self, text, position):
            label = QLabel(str(text), self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            label.setStyleSheet(self.LABEL_STYLE)
            label.adjustSize()
            self._chain_labels.append((label, tuple(position)))

        def plot_chains(
            self,
            chains_data,
            label_ids=None,
            preserve_view=False,
            waters=None,
            ions=None,
            chain_styles=None,
        ):
            self._clear_scene()
            if not chains_data:
                self.update()
                return

            label_ids = set(label_ids or ())
            chain_styles = chain_styles or {}
            chain_ids = sorted(chains_data.keys())
            should_refit = not preserve_view or self.max_range == 10.0
            if should_refit:
                calc_targets = (
                    label_ids
                    if self.center_on_labels and label_ids
                    else chain_ids
                )
                all_points = [
                    point
                    for chain_id in calc_targets
                    for point in chains_data.get(chain_id, ())
                ]
                if not all_points:
                    return
                points = np.asarray(all_points, dtype=float)
                center = points.mean(axis=0)
                self.global_center = tuple(center)
                distances = np.linalg.norm(points - center, axis=1)
                self.max_range = float(distances.max()) or 10.0

            for chain_id in chain_ids:
                coords = chains_data[chain_id]
                if not coords:
                    continue
                selected = chain_id in label_ids
                if chain_id in chain_styles:
                    color, line_width = chain_styles[chain_id]
                elif self.highlight_selected and selected:
                    color, line_width = (0.596, 0.765, 0.475, 0.9), 3.0
                else:
                    color = (1.0, 1.0, 1.0, 0.6 if not self.highlight_selected else 0.3)
                    line_width = 1.5 if not self.highlight_selected else 1.0
                line = ContextSafeLinePlotItem(
                    pos=np.asarray(coords, dtype=float),
                    color=color,
                    width=line_width,
                    antialias=True,
                    mode='line_strip',
                    glOptions='translucent',
                )
                self.addItem(line)
                if selected:
                    self._add_label(chain_id, coords[0])

            if waters:
                self.addItem(ContextSafeScatterPlotItem(
                    pos=np.asarray(waters, dtype=float),
                    color=(0.678, 0.847, 0.902, 0.8),
                    size=10.0,
                    pxMode=True,
                    glOptions='translucent',
                ))
            if ions:
                self.addItem(ContextSafeScatterPlotItem(
                    pos=np.asarray(ions, dtype=float),
                    color=(1.0, 0.714, 0.757, 0.9),
                    size=15.0,
                    pxMode=True,
                    glOptions='translucent',
                ))

            self.apply_zoom()


    class SoftwareChainCanvas(QWidget):
        """CPU-rendered 3D chain preview for XQuartz and other remote displays."""

        def __init__(
            self,
            parent=None,
            width=8,
            height=5,
            dpi=50,
            highlight_selected=False,
            center_on_labels=False,
            centered_labels=True,
        ):
            super().__init__(parent)
            self._preferred_size = QSize(
                max(1, round(width * dpi)),
                max(1, round(height * dpi)),
            )
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.setMouseTracking(True)
            self.highlight_selected = highlight_selected
            self.center_on_labels = center_on_labels
            self.centered_labels = centered_labels
            self.global_center = np.zeros(3, dtype=float)
            self.max_range = 10.0
            self.zoom_level = 50
            self.slider_callback = None
            self._chains = {}
            self._label_ids = set()
            self._waters = np.empty((0, 3), dtype=float)
            self._ions = np.empty((0, 3), dtype=float)
            self._chain_styles = {}
            self._rotation_x = math.radians(30.0)
            self._rotation_y = math.radians(-60.0)
            self._pan = np.zeros(2, dtype=float)
            self._last_mouse_pos = None

        def sizeHint(self):
            return QSize(self._preferred_size)

        def minimumSizeHint(self):
            return QSize(10, 10)

        @staticmethod
        def _qcolor(color):
            if isinstance(color, QColor):
                return QColor(color)
            values = tuple(color)
            if values and max(values) <= 1.0:
                values = tuple(round(value * 255) for value in values)
            if len(values) == 3:
                values += (255,)
            return QColor(*[max(0, min(255, int(value))) for value in values[:4]])

        def _rotation_matrix(self):
            cx, sx = math.cos(self._rotation_x), math.sin(self._rotation_x)
            cy, sy = math.cos(self._rotation_y), math.sin(self._rotation_y)
            rotate_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
            rotate_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
            return rotate_y @ rotate_x

        def _transform(self, points):
            points = np.asarray(points, dtype=float)
            if not len(points):
                return points.reshape((0, 3))
            return (points - self.global_center) @ self._rotation_matrix().T

        def _scale(self):
            zoom_divisor = max(0.05, 2.0 - (self.zoom_level / 100.0) * 1.9)
            extent = max(float(self.max_range), 0.001) * zoom_divisor
            return max(1.0, min(self.width(), self.height()) * 0.42 / extent)

        def _project(self, points):
            rotated = self._transform(points)
            scale = self._scale()
            projected = np.empty((len(rotated), 2), dtype=float)
            projected[:, 0] = self.width() / 2.0 + self._pan[0] + rotated[:, 0] * scale
            projected[:, 1] = self.height() / 2.0 + self._pan[1] - rotated[:, 1] * scale
            return projected, rotated[:, 2]

        def _screen_position(self, position):
            if self.width() <= 0 or self.height() <= 0:
                return None
            projected, _depth = self._project([position])
            x, y = projected[0]
            if not (-20 <= x <= self.width() + 20 and -20 <= y <= self.height() + 20):
                return None
            return float(x), float(y)

        def orbit(self, azimuth, elevation):
            self._rotation_y += math.radians(float(azimuth) * 0.5)
            self._rotation_x += math.radians(float(elevation) * 0.5)
            self.update()

        def pan(self, x, y, _z=0, relative="view"):
            self._pan += (float(x), float(y))
            self.update()

        def fit_structure(self):
            self.zoom_level = 50
            self._pan[:] = 0
            self.update()

        def set_zoom_from_slider(self, value):
            self.zoom_level = int(value)
            self.apply_zoom()

        def apply_zoom(self):
            self.update()

        def wheelEvent(self, event):
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta == 0:
                event.ignore()
                return
            self.zoom_level = max(1, min(100, self.zoom_level + (5 if delta > 0 else -5)))
            if self.slider_callback:
                self.slider_callback(self.zoom_level)
            self.update()
            event.accept()

        def mousePressEvent(self, event):
            self._last_mouse_pos = event.position().toPoint()
            event.accept()

        def mouseMoveEvent(self, event):
            current = event.position().toPoint()
            if self._last_mouse_pos is None:
                self._last_mouse_pos = current
            difference = current - self._last_mouse_pos
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.orbit(-difference.x(), difference.y())
            elif event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
                self.pan(difference.x(), difference.y())
            self._last_mouse_pos = current
            event.accept()

        def mouseReleaseEvent(self, event):
            self._last_mouse_pos = None
            event.accept()

        def plot_chains(
            self,
            chains_data,
            label_ids=None,
            preserve_view=False,
            waters=None,
            ions=None,
            chain_styles=None,
        ):
            self._chains = {
                chain_id: np.asarray(coords, dtype=float)
                for chain_id, coords in chains_data.items()
                if len(coords)
            }
            self._label_ids = set(label_ids or ())
            self._chain_styles = dict(chain_styles or {})
            self._waters = np.asarray(
                [] if waters is None else waters, dtype=float
            ).reshape((-1, 3))
            self._ions = np.asarray(
                [] if ions is None else ions, dtype=float
            ).reshape((-1, 3))

            should_refit = not preserve_view or self.max_range == 10.0
            if should_refit and self._chains:
                target_ids = (
                    self._label_ids
                    if self.center_on_labels and self._label_ids
                    else self._chains.keys()
                )
                point_sets = [
                    self._chains[chain_id]
                    for chain_id in target_ids
                    if chain_id in self._chains and len(self._chains[chain_id])
                ]
                if point_sets:
                    points = np.vstack(point_sets)
                    self.global_center = points.mean(axis=0)
                    distances = np.linalg.norm(points - self.global_center, axis=1)
                    self.max_range = float(distances.max()) or 10.0
                    self._pan[:] = 0
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#2C2C2C"))

            if not self._chains:
                painter.setPen(QColor("#8a8a8a"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No structure loaded")
                return

            chain_layers = []
            for chain_id, coords in self._chains.items():
                projected, depths = self._project(coords)
                chain_layers.append((float(depths.mean()), chain_id, projected))

            for _depth, chain_id, projected in sorted(chain_layers):
                selected = chain_id in self._label_ids
                if chain_id in self._chain_styles:
                    color, line_width = self._chain_styles[chain_id]
                elif self.highlight_selected and selected:
                    color, line_width = (0.596, 0.765, 0.475, 0.9), 3.0
                else:
                    color = (1.0, 1.0, 1.0, 0.6 if not self.highlight_selected else 0.3)
                    line_width = 1.5 if not self.highlight_selected else 1.0
                painter.setPen(QPen(self._qcolor(color), float(line_width)))
                for first, second in zip(projected, projected[1:]):
                    painter.drawLine(QPointF(*first), QPointF(*second))

                if selected and len(projected):
                    label_position = projected[0]
                    label_rect_x = label_position[0] - (12 if self.centered_labels else 0)
                    label_rect_y = label_position[1] - 20
                    painter.setPen(QColor("#111111"))
                    painter.setBrush(QBrush(QColor(245, 245, 245, 230)))
                    painter.drawRoundedRect(
                        int(label_rect_x), int(label_rect_y), 28, 18, 3, 3
                    )
                    painter.drawText(
                        int(label_rect_x), int(label_rect_y), 28, 18,
                        Qt.AlignmentFlag.AlignCenter, str(chain_id),
                    )

            painter.setPen(Qt.PenStyle.NoPen)
            for points, color, radius in (
                (self._waters, QColor(173, 216, 235, 210), 4.0),
                (self._ions, QColor(255, 182, 193, 230), 6.0),
            ):
                if not len(points):
                    continue
                painter.setBrush(QBrush(color))
                projected, _depths = self._project(points)
                for point in projected:
                    painter.drawEllipse(QPointF(*point), radius, radius)


    ChainCanvasBase = SoftwareChainCanvas if use_software_renderer else OpenGLChainCanvas


    class Mol3DCanvas(ChainCanvasBase):
        def __init__(self, parent=None, width=8, height=5, dpi=50):
            super().__init__(
                parent,
                width,
                height,
                dpi,
                highlight_selected=False,
                center_on_labels=True,
                centered_labels=True,
            )


    def generate_chain_id(index):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if index < len(chars): return chars[index]
        n = index - len(chars); base = len(chars)
        return chars[n // base] + chars[n % base]


    class TrimDialog(QDialog):
        def __init__(self, layers, full_cif_data, original_filename, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Trim Layers")
            self.resize(800, 600)
            self.layers = layers  
            self.full_cif_data = full_cif_data 
            self.original_filename = original_filename
            self.initUI()
            
            # Apply global theme styling directly from the active PyQt application
            qt_app = QApplication.instance()
            if qt_app:
                self.setStyleSheet(qt_app.styleSheet())

        def initUI(self):
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Select the layers you want to KEEP\nChains will be automatically renamed to maximum 2 letters upon saving to avoid truncations when opening/saving in Chimera(X)"))
            self.table = QTableWidget()
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["Keep", "Rank", "Original Chains", "Residues"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.populate_table()
            layout.addWidget(self.table)
            btn_layout = QHBoxLayout(); btn_layout.addStretch()
            self.btn_save = QPushButton("Save New CIF")
            self.btn_save.clicked.connect(self.save_cif)
            btn_layout.addWidget(self.btn_save)
            layout.addLayout(btn_layout)
            self.setLayout(layout)

        def update_data(self, layers, full_cif_data, filename):
            self.layers = layers
            self.full_cif_data = full_cif_data
            self.original_filename = filename
            self.populate_table()

        def populate_table(self):
            self.table.setRowCount(len(self.layers))
            self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Remove focus rectangle from table
            self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection) # Disable global selection highlighting

            for i, layer in enumerate(self.layers):
                # Column 0: Centered Checkbox Widget
                widget = QWidget()
                chk = QCheckBox()
                layout = QHBoxLayout(widget)
                layout.addWidget(chk)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                self.table.setCellWidget(i, 0, widget)

                # Column 1: Rank (Unclickable)
                rank_item = QTableWidgetItem(str(i + 1))
                rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                rank_item.setFlags(Qt.ItemFlag.ItemIsEnabled) 
                self.table.setItem(i, 1, rank_item)

                # Column 2: Chains (Unclickable)
                chains_item = QTableWidgetItem(", ".join(layer['chains']))
                chains_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, 2, chains_item)

                # Column 3: Residues (Unclickable)
                res_item = QTableWidgetItem(str(layer['residues']))
                res_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, 3, res_item)

        def save_cif(self):
            selected = []
            for i in range(self.table.rowCount()):
                # Fix: Retrieve the widget containing the checkbox, not the item
                widget = self.table.cellWidget(i, 0)
                checkbox = widget.findChild(QCheckBox)
                
                if checkbox and checkbox.isChecked():
                    for cid in self.layers[i]['chains']: 
                        selected.append(cid)
            
            if not selected:
                QMessageBox.warning(self, "Warning", "No layers selected.")
                return
            
            mapping = {old: generate_chain_id(idx) for idx, old in enumerate(selected)}
            name_part, ext = os.path.splitext(self.original_filename)
            
            path, _ = QFileDialog.getSaveFileName(self, "Save Trimmed CIF", f"{name_part}_trimmed.cif", "CIF Files (*.cif)")
            if path:
                try:
                    self.write_cif(path, mapping)
                    QMessageBox.information(self, "Success", f"Saved: {path}\n(Renamed {len(mapping)} chains)")
                    self.accept()
                except Exception as e: 
                    QMessageBox.critical(self, "Error", str(e))

        def write_cif(self, path, mapping):
            editor = StructureEditor(self.full_cif_data['source'])
            editor.trim(mapping, rename=mapping)
            editor.write(path, format='cif')


    class CIFMol3DCanvas(ChainCanvasBase):
        def __init__(self, parent=None, width=8, height=5, dpi=50):
            super().__init__(
                parent,
                width,
                height,
                dpi,
                highlight_selected=True,
                center_on_labels=False,
                centered_labels=False,
            )


    class CIFLayerIdentifier(QWidget):
        def __init__(self):
            super().__init__()
            self.created_temp_files = [] # Track specific files to delete
            
            # Ensure cleanup happens even if app crashes
            def cleanup_files():
                for f in self.created_temp_files:
                    if os.path.exists(f):
                        try: os.remove(f)
                        except: pass
            atexit.register(cleanup_files)
            
            self.working_file_path = None; self.layers_result = []; self.full_cif_data = {}
            self.initUI()
            qt_app = QApplication.instance()
            if qt_app: 
                self.setStyleSheet(qt_app.styleSheet())

        def closeEvent(self, e):
            # Cleanup temp files on close
            for f in self.created_temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            e.accept()

        def initUI(self):
            main_l = QHBoxLayout()
            splitter = QSplitter(Qt.Orientation.Horizontal)
            left = QWidget(); l_lay = QVBoxLayout(left); l_lay.setSpacing(10); l_lay.setContentsMargins(10,10,10,10)
            
            h_load = QHBoxLayout()
            self.btn_browse = QPushButton('Load PDB/CIF File')
            self.btn_browse.setIcon(create_outline_folder_icon())
            self.btn_browse.clicked.connect(self.browse_file)
            h_load.addWidget(self.btn_browse)
            self.lbl_status = QLabel("No file loaded"); self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
            h_load.addWidget(self.lbl_status); l_lay.addLayout(h_load)
            
            self.grp_ops = QGroupBox("Operations"); self.grp_ops.setEnabled(False)
            ops_l = QVBoxLayout()
            self.btn_trim = QPushButton("Trim Best Layers")
            self.btn_trim.setToolTip("Select the layers wanted and save as a new CIF file")
            self.btn_trim.clicked.connect(self.open_trim_dialog)
            ops_l.addWidget(self.btn_trim)
            self.grp_ops.setLayout(ops_l); l_lay.addWidget(self.grp_ops)
            
            grp_p = QGroupBox("Associated Molecules"); form = QFormLayout()
            self.ed_chn = QLineEdit("4"); self.ed_chn.setValidator(QIntValidator())
            form.addRow("Chains per Layer:", self.ed_chn)
            self.ed_chn.setToolTip("Number of protofilaments you have in each layer")
            self.ed_zmn = QLineEdit("0.0"); self.ed_zmn.setValidator(QDoubleValidator())
            form.addRow("Z Shift Min (Å):", self.ed_zmn)
            self.ed_zmn.setToolTip("Within a layer, the minimum shift in Z direction that a chain can be\nUsually 0Å")
            self.ed_zmx = QLineEdit("4.0"); self.ed_zmx.setValidator(QDoubleValidator())
            form.addRow("Z Shift Max (Å):", self.ed_zmx)
            self.ed_zmx.setToolTip("Within a layer, the maximum shift in Z direction that a chain can be\nCan NOT be larger than the thickness of a typical layer")
            self.ed_chn.editingFinished.connect(self.recalc); self.ed_zmn.editingFinished.connect(self.recalc); self.ed_zmx.editingFinished.connect(self.recalc)
            grp_p.setLayout(form); l_lay.addWidget(grp_p)
            
            self.txt = QTextBrowser(); self.txt.setOpenLinks(False); self.txt.anchorClicked.connect(self.link_clk)
            self.txt.setStyleSheet("font-family: Consolas, monospace;")
            l_lay.addWidget(self.txt)
            
            right = QWidget(); r_lay = QHBoxLayout(right); r_lay.setContentsMargins(0,0,0,0)
            self.cvs = CIFMol3DCanvas(self, dpi=100); r_lay.addWidget(self.cvs, 1)
            sl = QSlider(Qt.Orientation.Vertical); sl.setRange(1, 100); sl.setValue(50); sl.valueChanged.connect(lambda v: self.cvs.set_zoom_from_slider(v))
            self.cvs.slider_callback = lambda v: sl.setValue(v)
            r_lay.addWidget(sl)
            
            splitter.addWidget(left); splitter.addWidget(right)
            splitter.setSizes([600, 750]) # Force Left Panel to 500px width
            splitter.setStretchFactor(0, 4); splitter.setStretchFactor(1, 6)
            main_l.addWidget(splitter); self.setLayout(main_l)

        def recalc(self):
            if self.working_file_path: QTimer.singleShot(10, lambda: self.process_cif(self.working_file_path))
        
        def browse_file(self):
            f, _ = QFileDialog.getOpenFileName(self, "Open PDB/CIF File", "", "Model Files (*.pdb *.cif *.mmcif);;CIF Files (*.cif *.mmcif);;PDB Files (*.pdb);;All Files (*)")
            if f:
                self.load_cif(f)

        def load_cif(self, filename):
            self.loaded_filename = os.path.basename(filename)
            self.lbl_status.setText(f"Loaded: {self.loaded_filename}")
            
            # Create temp file in the SAME folder with unique ID
            folder = os.path.dirname(filename)
            unique_suffix = uuid.uuid4().hex[:8]
            dst = os.path.join(folder, f"temp_work_{unique_suffix}{os.path.splitext(filename)[1]}")
            
            shutil.copy2(filename, dst)
            self.created_temp_files.append(dst) # Track for deletion
            
            self.working_file_path = dst
            self.grp_ops.setEnabled(True)
            self.txt.clear(); self.txt.append(f"Loaded {filename}")
            self.process_cif(dst)

        def link_clk(self, url):
            try:
                idx = int(url.toString())
                if 0 <= idx < len(self.layers_result):
                    self.cvs.plot_chains(self.chains_data_plot, label_ids=set(self.layers_result[idx]['chains']), preserve_view=True, waters=getattr(self, 'current_waters', []), ions=getattr(self, 'current_ions', []))
            except: pass

        def open_trim_dialog(self):
            if not self.layers_result: return
            
            # Check if dialog is already open; if so, bring to front
            if hasattr(self, 'trim_dialog') and self.trim_dialog and self.trim_dialog.isVisible():
                self.trim_dialog.raise_()
                self.trim_dialog.activateWindow()
                return

            # Open as a non-modal window (allows interaction with main window)
            self.trim_dialog = TrimDialog(self.layers_result, self.full_cif_data, self.loaded_filename, self)
            self.trim_dialog.show()

        def parse_cif(self, path):
            editor = StructureEditor(path)
            chains, waters, ions = editor.preview()
            return {c: [r['coord'] for r in rows] for c, rows in chains.items()}, waters, ions, {'source': path}

        def process_cif(self, path):
            chn_max = int(self.ed_chn.text()); zmn = float(self.ed_zmn.text()); zmx = float(self.ed_zmx.text())
            try:
                self.txt.append("Parsing..."); coords, waters, ions, self.full_cif_data = self.parse_cif(path)
                if not coords: self.txt.append("No CA atoms found."); return
                self.chains_data_plot = {c: [tuple(p) for p in coords[c]] for c in coords}
                self.current_waters = waters
                self.current_ions = ions
                cents = {c: np.mean(coords[c], axis=0) for c in coords}
                pool = sorted(list(coords.keys()), key=lambda c: cents[c][2])
                self.layers_result = []
                while pool:
                    ref = pool.pop(0); cur = [ref]; rz = cents[ref][2]
                    cands = sorted([(abs(cents[c][2]-rz), c) for c in pool if zmn <= abs(cents[c][2]-rz) <= zmx], key=lambda x:x[0])
                    sel = [c[1] for c in cands[:chn_max-1]]
                    cur.extend(sel); 
                    for c in sel: pool.remove(c)
                    res_cnt = sum(len(coords[c]) for c in cur)
                    self.layers_result.append({'chains': cur, 'count': len(cur), 'residues': res_cnt})
                self.layers_result.sort(key=lambda x: (x['count'], x['residues']), reverse=True)
                
                h_lines = []
                high_ids = set()
                for i, lay in enumerate(self.layers_result):
                    lnk = f"<a href='{i}' style='color: #97c379; font-weight: bold; text-decoration: none;'>Layer {i+1}</a>"
                    h_lines.append(f"{lnk}: {', '.join(lay['chains'])} [Chains, Residues]:{lay['count']}, {lay['residues']}<br>")
                    if i==0: high_ids.update(lay['chains'])
                self.txt.setHtml("<br>".join(h_lines))
                self.cvs.plot_chains(self.chains_data_plot, label_ids=high_ids, waters=waters, ions=ions)
                
                # Update Trim Dialog if it is open
                if hasattr(self, 'trim_dialog') and self.trim_dialog and self.trim_dialog.isVisible():
                    self.trim_dialog.update_data(self.layers_result, self.full_cif_data, self.loaded_filename)

            except Exception as e: self.txt.append(str(e))


    class PDBLayerIdentifier(QWidget):
        
        CHAIN_RECORD_SPECS = {
            "DBREF":  [(11, 13)], "DBREF1": [(11, 13)], "DBREF2": [(11, 13)],
            "SEQADV": [(15, 17)], "MODRES": [(15, 17)], "SEQRES": [(10, 12)],
            "HET":    [(11, 13)], "SSBOND": [(14, 16), (28, 30)], "CISPEP": [(14, 16), (28, 30)],
            "LINK":   [(20, 22), (50, 52)], "SITE":   [(21, 23), (32, 34), (43, 45), (53, 56)],
            "ATOM":   [(20, 22)], "ANISOU": [(20, 22)], "TER":    [(20, 22)],
            "HETATM": [(20, 22)], "HELIX":  [(18, 20), (30, 32)], "SHEET":  [(20, 22), (31, 33)]
        }

        def __init__(self):
            super().__init__()
            self.final_sandwiches = [] 
            self.detected_layers = 0
            self.detected_units = 0
            self.layers_per_unit = 1
            self.detection_result = None
            self.results = {}
            self.created_temp_files = [] # Track specific files
            
            # Ensure cleanup happens even if app crashes
            def cleanup_files():
                for f in self.created_temp_files:
                    if os.path.exists(f):
                        try: os.remove(f)
                        except: pass
            atexit.register(cleanup_files)

            self.working_file_path = None 
            self.original_filename_display = ""
            self.initUI()

        def closeEvent(self, event):
            # Cleanup temp files on close
            for f in self.created_temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            event.accept()

        def initUI(self):
            main_layout = QHBoxLayout()
            splitter = QSplitter(Qt.Orientation.Horizontal)

            # ================= LEFT PANE =================
            left_widget = QWidget()
            layout = QVBoxLayout(left_widget)
            layout.setSpacing(10)
            layout.setContentsMargins(10, 10, 10, 10)

            hb_load = QHBoxLayout()
            self.btn_browse = QPushButton('Load PDB/CIF File')
            self.btn_browse.setIcon(create_outline_folder_icon())
            self.btn_browse.clicked.connect(self.browse_file)
            hb_load.addWidget(self.btn_browse)
            
            self.lbl_status = QLabel("No file loaded")
            self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
            hb_load.addWidget(self.lbl_status)

            # Add a stretch to act like a spring, pushing the checkbox to the far right
            hb_load.addStretch()

            self.lbl_orientation = QLabel("Undetermined")
            hb_load.addWidget(self.lbl_orientation)

            layout.addLayout(hb_load)

            self.grp_ops = QGroupBox("Operations (Applied to Temporary Copy)")
            self.grp_ops.setEnabled(False) 
            v_ops = QVBoxLayout()

            hb_trim = QHBoxLayout()
            self.btn_trim = QPushButton("Trim/Expand Layers")
            self.btn_trim.setToolTip("Trim each protofilament to at most the target number of layers, or expand to the target")
            self.btn_trim.clicked.connect(self.action_trim)
            hb_trim.addWidget(self.btn_trim)
            hb_trim.addWidget(QLabel("Target Layers:"))
            self.edit_trim = QLineEdit("5")
            self.edit_trim.setValidator(QIntValidator(1, 500))
            self.edit_trim.setFixedWidth(50)
            hb_trim.addWidget(self.edit_trim)
            hb_trim.addStretch()
            v_ops.addLayout(hb_trim)

            hb_pf_trim = QHBoxLayout()
            self.btn_remove_pf = QPushButton("Trim Protofilament")
            self.btn_remove_pf.setToolTip("Trim away the selected protein protofilament or unattached ligand site")
            self.btn_remove_pf.clicked.connect(self.action_remove_pf)
            hb_pf_trim.addWidget(self.btn_remove_pf)
            self.cmb_protofilaments = QComboBox()
            hb_pf_trim.addWidget(self.cmb_protofilaments)
            v_ops.addLayout(hb_pf_trim)

            hb_ren = QHBoxLayout()
            self.btn_rename = QPushButton("Rename Chains")
            self.btn_rename.setToolTip("Rename the chain IDs")
            self.btn_rename.clicked.connect(self.action_rename)
            hb_ren.addWidget(self.btn_rename)
            hb_ren.addWidget(QLabel("(Rename Chain ID from the Middle Out)"))
            v_ops.addLayout(hb_ren)

            # --- NEW: Evaluate Beta Sheet ---
            self.btn_beta = QPushButton("Re-evaluate β-sheet structure")
            self.btn_beta.setToolTip("Re-evaluate β-sheet structure and annote to the PDB file\nUsing the gold-standard DSSP algorithm in Chimera\nMUST be greater than 1 layer")
            self.btn_beta.clicked.connect(self.action_evaluate_beta)
            self.btn_beta.setEnabled(False) 
            v_ops.addWidget(self.btn_beta)
            
            # Row 2: Checkbox (Added directly to vertical layout to sit on the next line)
            self.chk_numbers_first = QCheckBox("Use Numbers (0-9) after Z")
            self.chk_numbers_first.setToolTip("Use 0-9 before moving to double letters (AA)\nUseful for older software like Chimera no X\nYou can rename again without numbers after you done molmapping your final models in Chimera when making FSC curves\nYou don't need this if you have less than 5] protofilaments in a single layer\nAlso there is no difference if you have more than [7 protofilaments in a single layer")
            v_ops.addWidget(self.chk_numbers_first)

            self.chk_only_numbers = QCheckBox("Use only numbers")
            self.chk_only_numbers.setToolTip("Rename chains using strictly integers (0, 1, ... 99)")
            v_ops.addWidget(self.chk_only_numbers)

            # Logic to auto de-select the other option
            self.chk_numbers_first.toggled.connect(lambda state: state and self.chk_only_numbers.setChecked(False))
            self.chk_only_numbers.toggled.connect(lambda state: state and self.chk_numbers_first.setChecked(False))

            hb_res = QHBoxLayout()
            self.btn_restrain = QPushButton("Generate Restrain File")
            self.btn_restrain.setToolTip("Generate the restrain torsion file for the current working copy")
            self.btn_restrain.clicked.connect(self.action_restrain)
            hb_res.addWidget(self.btn_restrain)
            hb_res.addWidget(QLabel("(Export .cxc File)"))
            v_ops.addLayout(hb_res)

            hb_conn = QHBoxLayout()
            self.btn_connect = QPushButton("Connect Chains")
            self.btn_connect.setToolTip("This feature helps to merge all chains into one")
            self.btn_connect.clicked.connect(self.action_connect_chain)
            hb_conn.addWidget(self.btn_connect)
            hb_conn.addWidget(QLabel("(Merge all into one chain)"))
            v_ops.addLayout(hb_conn)

            self.grp_ops.setLayout(v_ops)
            layout.addWidget(self.grp_ops)

            self.btn_save = QPushButton("Save Final PDB/CIF")
            self.btn_save.setEnabled(False)
            self.btn_save.clicked.connect(self.action_save_final)
            layout.addWidget(self.btn_save)

            # --- START: Detection Parameters with Tooltips ---
            self.grp_params = QGroupBox("Associated Molecules")
            params_layout = QFormLayout()
            params_layout.setContentsMargins(5, 5, 5, 5)

            # Helper to create text inputs with tooltips
            def create_param_input(default_val, tooltip_msg, is_int=False):
                line_edit = QLineEdit()
                line_edit.setText(str(default_val))
                if is_int:
                    line_edit.setValidator(QIntValidator())
                else:
                    line_edit.setValidator(QDoubleValidator())
                line_edit.editingFinished.connect(self.on_param_change) # Auto-reload
                line_edit.setToolTip(tooltip_msg) # Show explanation on hover
                return line_edit

            self.edit_xy_limit = QLineEdit("3.0")  # Compatibility with companion selection.
            self._water_z_auto = True
            self.edit_water_z = create_param_input(4.0, "")
            self.edit_water_z.validator().setBottom(0.0)
            self.edit_water_z.textEdited.connect(self.on_water_limit_edited)
            self.btn_water_auto = QPushButton("Auto")
            self.btn_water_auto.setStyleSheet('QPushButton { padding: 2px 6px; }')
            self.btn_water_auto.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.btn_water_auto.clicked.connect(self.apply_auto_water_limit)
            water_row = QHBoxLayout()
            water_row.setContentsMargins(0, 0, 0, 0)
            water_row.setSpacing(4)
            water_row.addWidget(self.edit_water_z, 1)
            water_row.addWidget(self.btn_water_auto)
            params_layout.addRow("Water/Ion Axial Limit (Å):", water_row)
            self.update_water_axial_limit()


            self.grp_params.setLayout(params_layout)
            layout.addWidget(self.grp_params)
            # --- END: Detection Parameters ---

            self.text_area = QTextEdit()
            self.text_area.setReadOnly(True)
            font = self.text_area.font()
            font.setFamily("Courier New")
            font.setPointSize(9)
            self.text_area.setFont(font)
            layout.addWidget(self.text_area)

            # ================= RIGHT PANE =================
            right_widget = QWidget()
            r_layout = QHBoxLayout(right_widget)
            r_layout.setContentsMargins(0,0,0,0)
            r_layout.setSpacing(2)

            # 1. Canvas
            self.mol_canvas = Mol3DCanvas(self, width=8, height=5, dpi=100)
            r_layout.addWidget(self.mol_canvas, 1)

            # 2. Zoom Slider
            self.zoom_slider = QSlider(Qt.Orientation.Vertical)
            self.zoom_slider.setRange(1, 100)
            self.zoom_slider.setValue(50)
            self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksRight)
            self.zoom_slider.setTickInterval(10)
            self.zoom_slider.valueChanged.connect(self.on_slider_change)
            r_layout.addWidget(self.zoom_slider)

            self.mol_canvas.slider_callback = self.update_slider_visual

            splitter.addWidget(left_widget)
            splitter.addWidget(right_widget)
            splitter.setStretchFactor(0, 4) 
            splitter.setStretchFactor(1, 6) 

            main_layout.addWidget(splitter)
            self.setLayout(main_layout)

        def on_slider_change(self, value):
            self.mol_canvas.set_zoom_from_slider(value)

        def update_slider_visual(self, value):
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(value)
            self.zoom_slider.blockSignals(False)
        
        def on_water_limit_edited(self, text):
            self._water_z_auto = False
            self.update_water_axial_limit()


        def apply_auto_water_limit(self):
            self._water_z_auto = True
            self.update_water_axial_limit()
            self.on_param_change()


        def update_water_axial_limit(self):
            limit = automatic_water_axial_limit(self.detection_result)
            if self._water_z_auto and limit is not None:
                # The displayed number is also the value used by editing operations.
                self.edit_water_z.setText(f'{limit:.3f}')
            tip = ('Maximum axial distance from protein-chain centers for associated molecules.\n'
                   'Suggested: half the detected layer spacing + 10% fluctuation tolerance '
                   '(layer spacing × 0.5 × 1.10).\n'
                   'Auto uses the median physical layer spacing across protofilaments. '
                   'Edit the value to override it; click Auto to recalculate and reapply.')
            if limit is None:
                tip += '\nLayer spacing is unavailable; the displayed value is kept (initial fallback: 4.0 Å).'
            else:
                tip += f'\nDetected spacing: {limit / 0.55:.3f} Å; suggested limit: {limit:.3f} Å.'
            tip += '\nMode: ' + ('Automatic' if self._water_z_auto else 'Manual') + '.'
            self.edit_water_z.setToolTip(tip)
            self.btn_water_auto.setEnabled(limit is not None)
            self.btn_water_auto.setToolTip(tip)


        def on_param_change(self):
            # Only run if a file has been loaded
            if self.working_file_path:
                self.text_area.append("\n--- Parameters changed: Re-calculating ---")
                # Add a small delay or just run it directly
                QTimer.singleShot(10, lambda: self.process_pdb(self.working_file_path))

        def get_param(self, widget, default):
            try:
                val = float(widget.text())
                return val
            except ValueError:
                return default
        
        def get_kept_heteroatoms(self, keep_layer_indices, xy_limit, water_z_limit, *, keep_chains=None):
            if not self.working_file_path or not self.final_sandwiches:
                return set()
            editor = StructureEditor(self.working_file_path)
            axis = (getattr(self, 'detection_result', None) or {}).get('axis')
            return editor.associated_residues_for_layers(self.final_sandwiches, keep_layer_indices, axis,
                                                        water_z_limit, keep_chains=keep_chains)

        # ================= LOGIC: FILE HANDLING =================

        def browse_file(self, filename=None):
            if not filename or isinstance(filename, bool):
                
                filename, _ = QFileDialog.getOpenFileName(self, "Open PDB/CIF File", "", 
                                                        "Model Files (*.pdb *.cif *.mmcif);;PDB Files (*.pdb);;CIF Files (*.cif *.mmcif);;All Files (*)")
            if filename:
                self.original_filename_display = os.path.basename(filename)
                self.lbl_status.setText(f"Loaded: {self.original_filename_display}")
                
                # Create temp file in the SAME folder with unique ID
                folder = os.path.dirname(filename)
                unique_suffix = uuid.uuid4().hex[:8]
                _, ext = os.path.splitext(filename)
                temp_path = os.path.join(folder, f"temp_working_{unique_suffix}{ext}")
                
                shutil.copy2(filename, temp_path)
                self.created_temp_files.append(temp_path) # Track it
                
                self.working_file_path = temp_path
                
                self.grp_ops.setEnabled(True)
                self.btn_save.setEnabled(True)
                
                self.text_area.clear()
                self.text_area.append(f"Loaded {filename}")

                self._axis_reference = None
                self._shown_source_notices = set()
                self._water_z_auto = True
                self.edit_water_z.setText('4.0')

                self.process_pdb(self.working_file_path)

        def action_rename(self):
            if not self.working_file_path: return
            try:
                old_sandwiches = self.final_sandwiches
                mapping = self.create_renaming_mapping(old_sandwiches)
                
                folder = os.path.dirname(self.working_file_path)
                unique_suffix = uuid.uuid4().hex[:8]
                ext = ".cif" if self.working_file_path.lower().endswith(('.cif', '.mmcif')) else ".pdb"
                new_temp = os.path.join(folder, f"temp_renamed_{unique_suffix}{ext}")
                self.created_temp_files.append(new_temp)
                
                if self.working_file_path.lower().endswith(('.cif', '.mmcif')):
                    self.write_renamed_cif(self.working_file_path, new_temp, self.final_sandwiches)
                else:
                    self.write_renamed_pdb(self.working_file_path, new_temp, self.final_sandwiches)
                
                self._adopt_working_file(new_temp, mapping)
                self.text_area.append("\n>>>> Applied Renaming")
                self.process_pdb(self.working_file_path, old_sandwiches=old_sandwiches, rename_map=mapping)
            except Exception as e: QMessageBox.critical(self, "Error", f"Rename failed: {e}")

        class ExpandDialog(QDialog):
            def __init__(self, current_layers, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Expand Layers")
                self.setModal(False)
                self.setWindowModality(Qt.WindowModality.NonModal)
                self.current_layers = current_layers
                self.unit_layers = getattr(parent, 'layers_per_unit', 1) or 1
                detection = getattr(parent, 'detection_result', None) or {}
                self.suggested_repeat = detection.get('suggested_repeat_units', 2 if detection.get('antiparallel') or detection.get('parent_antiparallel') else 1)
                self.compatible_repeats = detection.get('compatible_repeat_units')
                self.terminal_completion = terminal_completion_supported(detection)
                self.completion_count = sum(current_layers - t['units'] for t in detection.get('protofilaments', [])) if self.terminal_completion else 0
                rises = [track['unit_rise'] for track in detection.get('protofilaments', [])
                         if track.get('unit_rise') is not None]
                self.manual_rise = float(np.median(rises)) if rises else 4.8 * self.unit_layers
                self.options = None
                self.initUI()

            def initUI(self):
                layout = QVBoxLayout(self)
                layout.setContentsMargins(12, 12, 12, 12)
                layout.setSpacing(10)
                if self.terminal_completion:
                    note = QLabel(f"Complete shorter protofilaments first: add {self.completion_count} modeled terminal chains so every stack has {self.current_layers} layers.\n"
                                  f"Completion uses the measured {self.suggested_repeat}-chain repeat and preserves existing atoms and strand orientations. The end is chosen to balance the shorter stack against the longer stacks.\n"
                                  "Then expand the completed assembly using the settings below. Added chains and ends will be listed in the log.")
                    note.setWordWrap(True)
                    layout.addWidget(note)
                self.chk_auto = QCheckBox("Compute Twist/Rise")
                self.chk_auto.setStyleSheet("QCheckBox:disabled { color: #808080; } QCheckBox::indicator:disabled { border-color: #555; background-color: #303030; }")
                self.chk_auto.setChecked(True)
                # The enabled row also supplies the tooltip when its checkbox is disabled.
                self.auto_row = QWidget()
                auto_layout = QHBoxLayout(self.auto_row)
                auto_layout.setContentsMargins(0, 0, 0, 0)
                auto_layout.addWidget(self.chk_auto)
                layout.addWidget(self.auto_row)

                period_form = QFormLayout()
                self.edit_period = QLineEdit(str(self.suggested_repeat))
                self.edit_period.setMinimumWidth(150)
                self.edit_period.setToolTip("1: A-A-A; 2: A-B-A-B; 3: A-B-C-A-B-C. Zero or any negative whole number: random compatible source conformations.")
                period_form.addRow("Alternating period (chain units):", self.edit_period)
                layout.addLayout(period_form)

                self.chk_compute_axis = QCheckBox("Compute Helical Axis")
                self.chk_compute_axis.setToolTip("Checked: derive the axis from source repeats. With only one source unit, estimate the layer normal from C-alpha PCA (the smallest-eigenvalue direction, perpendicular to the broadest PCA projection). This shape estimate cannot determine a true repeat axis or its polarity. Unchecked: use the global Z axis; align the structure with Z first.")
                self.chk_compute_axis.setChecked(self.current_layers > 1)
                layout.addWidget(self.chk_compute_axis)

                form_layout = QFormLayout()
                self.edit_twist = QLineEdit("0.000")
                self.edit_rise = QLineEdit(f"{self.manual_rise:.3f}")
                self.edit_twist.setPlaceholderText("Automatic")
                self.edit_rise.setPlaceholderText("Automatic")
                self.edit_rise.setToolTip("Manual starting value: detected chain-unit rise, or 4.8 Å per layer when no repeat was detected.")
                self._showing_auto = False
                form_layout.addRow("Twist per chain unit (°):", self.edit_twist)
                form_layout.addRow("Rise per chain unit (Å):", self.edit_rise)
                layout.addLayout(form_layout)

                self.parameter_note = QLabel()
                self.parameter_note.setWordWrap(True)
                self.parameter_note.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                layout.addWidget(self.parameter_note)
                self.availability_note = QLabel()
                self.availability_note.setWordWrap(True)
                self.availability_note.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                layout.addWidget(self.availability_note)

                self.btn_expand = QPushButton("Complete && Expand" if self.terminal_completion else "Expand")
                self.btn_expand.clicked.connect(self.accept)
                layout.addWidget(self.btn_expand)
                self.setMinimumWidth(max(520, self.fontMetrics().horizontalAdvance("Alternating period (chain units):") + 210))
                self.chk_auto.stateChanged.connect(self.toggle_manual)
                self.chk_compute_axis.stateChanged.connect(self.toggle_manual)
                self.edit_period.textChanged.connect(self.toggle_manual)
                self.toggle_manual()

            def alternating_period(self):
                try:
                    return int(self.edit_period.text().strip())
                except ValueError:
                    return None

            def toggle_manual(self):
                period = self.alternating_period()
                randomize = period is not None and period <= 0
                effective = self.suggested_repeat if randomize else period
                compatible = self.compatible_repeats is None or effective in self.compatible_repeats
                valid = effective is not None and 0 < effective <= self.current_layers and compatible
                auto_available = valid and self.current_layers > effective
                auto_tip = ('Fit twist and rise per chain unit from equivalent positions in the source repeat. '
                            'When unchecked, use the entered twist and rise. ')
                if self.current_layers <= 1:
                    auto_tip += 'Unavailable: only one source chain unit per stack is present; at least two are needed to measure twist/rise.'
                elif not valid:
                    auto_tip += 'Unavailable: enter a compatible alternating period supported by the source units.'
                elif not auto_available:
                    auto_tip += f'Unavailable: a {effective}-unit repeat needs at least {effective + 1} source units to measure twist/rise.'
                else:
                    auto_tip += f'The selected {effective}-unit repeat needs at least {effective + 1} source units.'
                self.chk_auto.setToolTip(auto_tip)
                self.auto_row.setToolTip(auto_tip)
                self.chk_auto.setEnabled(auto_available)
                if not auto_available and self.chk_auto.isChecked():
                    self.chk_auto.blockSignals(True)
                    self.chk_auto.setChecked(False)
                    self.chk_auto.blockSignals(False)
                self.btn_expand.setEnabled(valid)
                is_auto = self.chk_auto.isChecked()
                if is_auto and not self._showing_auto:
                    self._manual_values = (self.edit_twist.text(), self.edit_rise.text())
                    self.edit_twist.clear()
                    self.edit_rise.clear()
                elif not is_auto and self._showing_auto:
                    self.edit_twist.setText(self._manual_values[0])
                    self.edit_rise.setText(self._manual_values[1])
                self._showing_auto = is_auto
                self.edit_twist.setEnabled(not is_auto)
                self.edit_rise.setEnabled(not is_auto)
                layer_word = 'layer' if self.unit_layers == 1 else 'layers'
                if randomize:
                    pattern = 'Random compatible source chain units\nProtein identities and strand directions remain in order.'
                elif period is not None:
                    unit_word = 'unit' if period == 1 else 'units'
                    pattern = f'Repeat every {period} chain {unit_word} ({period * self.unit_layers} layers)'
                else:
                    pattern = 'Enter a whole number; 0 or any negative number selects random.'
                axis_note = ''
                if self.current_layers <= 1:
                    axis_note = ('\n\nAxis: estimated from the single-unit shape (C-alpha PCA plane normal). '
                                 'Repeat direction and polarity cannot be measured from one unit.'
                                 if self.chk_compute_axis.isChecked() else '\n\nAxis: global Z.')
                self.parameter_note.setText(
                    f'Current structure:\nOne chain unit spans {self.unit_layers} physical {layer_word}\n\n'
                    f'Alternating pattern:\n{pattern}{axis_note}')
                availability = ''
                if effective is not None and effective > self.current_layers:
                    availability = f'At least {effective} source units are needed to include every template.'
                elif effective is not None and not compatible:
                    availability = 'This period would exchange different proteins or opposite strand orientations. Choose a compatible alternating period.'
                elif not auto_available and valid:
                    availability = f'Compute Twist/Rise needs at least {effective + 1} source units. Enter manual twist and rise for this fragment.'
                self.availability_note.setText(availability)
                self.availability_note.setVisible(bool(availability))
                self.edit_twist.setStyleSheet("color: gray;" if is_auto else "")
                self.edit_rise.setStyleSheet("color: gray;" if is_auto else "")
                self._fit_text()
                QTimer.singleShot(0, self._fit_text)

            def _fit_text(self):
                # Reserve wrapped text height under the current application theme.
                text_width = self.minimumWidth() - 24
                for label in (self.parameter_note, self.availability_note):
                    label.setMinimumHeight(max(0, label.heightForWidth(text_width)) if label.text() else 0)
                self.layout().activate()
                height = max(self.sizeHint().height(), self.minimumSizeHint().height())
                self.setMinimumHeight(height)
                self.resize(max(self.width(), self.minimumWidth()), max(self.height(), height))

            def accept(self):
                if not self.btn_expand.isEnabled():
                    return
                automatic = self.chk_auto.isChecked()
                twist = rise = 0.0
                if not automatic:
                    try:
                        twist, rise = float(self.edit_twist.text()), float(self.edit_rise.text())
                        if not math.isfinite(twist) or not math.isfinite(rise) or abs(rise) < 1e-8:
                            raise ValueError()
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Input", "Enter finite twist/rise values, with a nonzero rise.")
                        return
                self.options = dict(use_auto=automatic, repeat_units=self.alternating_period(),
                                    use_computed_axis=self.chk_compute_axis.isChecked(),
                                    manual_twist=twist, manual_rise=rise)
                super().accept()

        def action_remove_pf(self):
            if not self.working_file_path or not self.final_sandwiches: return
            if self.cmb_protofilaments.count() == 0: return
            
            try:
                target = self.cmb_protofilaments.currentData()
                if target is None:
                    return
                if target['kind'] == 'ligand':
                    editor = StructureEditor(self.working_file_path)
                    trim_ligand_target(editor, target)
                    fmt = 'cif' if self.working_file_path.lower().endswith(('.cif', '.mmcif')) else 'pdb'
                    output = os.path.join(os.path.dirname(self.working_file_path),
                                          f'temp_trimmed_ligand_{uuid.uuid4().hex[:8]}.{fmt}')
                    self.created_temp_files.append(output)
                    editor.write(output, format=fmt)
                    self._adopt_working_file(output)
                    self.operation_log('Trimmed ' + target['label'])
                    self.process_pdb(output)
                    return
                target_chains = {key[0] for key in target['residues']}
                pf_index = next((i for i, stack in enumerate(self.final_sandwiches)
                                 if target_chains.intersection(stack)), None)
                if pf_index is None or pf_index < 0 or pf_index >= len(self.final_sandwiches): return
                keep_chains = set()
                
                for i, sandwich in enumerate(self.final_sandwiches):
                    if i != pf_index:
                        for chain_id in sandwich:
                            keep_chains.add(chain_id)
                            
                if not keep_chains:
                    QMessageBox.warning(self, "Warning", "Cannot remove the last remaining protofilament.")
                    return
                
                mid_chain = self.final_sandwiches[pf_index][len(self.final_sandwiches[pf_index])//2]
                self.text_area.append(f"Removing Protofilament with middle chain {mid_chain}...")
                
                folder = os.path.dirname(self.working_file_path)
                unique_suffix = uuid.uuid4().hex[:8]
                ext = ".cif" if self.working_file_path.lower().endswith(('.cif', '.mmcif')) else ".pdb"
                new_temp = os.path.join(folder, f"temp_trimmed_pf_{unique_suffix}{ext}")
                self.created_temp_files.append(new_temp)
                
                water_z_limit = self.get_param(self.edit_water_z, 4.0)
                editor = StructureEditor(self.working_file_path)
                core = {c for stack in self.final_sandwiches for c in stack}
                axis = (self.detection_result or {}).get('axis')
                owners = editor.associated_residues(core, water_z_limit, axis)
                owners.update(editor.covalent_residue_owners())
                keep_het_lines = {residue for residue, owner in owners.items() if owner in keep_chains}
                
                if self.working_file_path.lower().endswith(('.cif', '.mmcif')):
                    self.write_trimmed_cif(self.working_file_path, new_temp, keep_chains, keep_het_lines)
                else:
                    self.write_trimmed_pdb(self.working_file_path, new_temp, keep_chains, keep_het_lines)
                    
                self._adopt_working_file(new_temp)
                self.text_area.append("\n>>>> Applied Protofilament Trimming")
                self.process_pdb(self.working_file_path)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Action failed: {e}")

        def action_trim(self):
            if not self.working_file_path or not self.final_sandwiches:
                return
            try:
                target_layers = int(self.edit_trim.text())
                if target_layers <= 0:
                    raise ValueError()
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Enter a positive integer for Target Layers.")
                return
            needs_completion = not self.detection_result['complete']
            if target_layers == self.detected_layers and not needs_completion:
                return
            if target_layers < self.detected_layers and (self.layers_per_unit is None or
                    any(not track['complete'] for track in self.detection_result['protofilaments'])):
                QMessageBox.warning(self, "Incomplete stacks", "Layer trimming requires contiguous protofilaments with the same number of layers per chain. Unequal protofilament lengths are supported; internal gaps and inconsistent layer multiplicity must be resolved first.")
                return
            if target_layers >= self.detected_layers and needs_completion and not terminal_completion_supported(self.detection_result):
                QMessageBox.warning(self, "Incomplete stacks", "Automatic terminal completion requires contiguous stacks with a supported orientation motif and one layer per chain. Parallel stacks need at least two chains and may lack up to two terminal chains; other supported motifs need at least three chains and may lack one. Choose a target at least as large as the longest stack. Internal gaps and ambiguous stacks cannot be completed automatically.")
                return
            unit_layers = self.layers_per_unit or 1
            if target_layers % unit_layers:
                QMessageBox.warning(self, "Whole chain units required", f"Each chain forms {unit_layers} layers. Choose a multiple of {unit_layers}; chains remain intact.")
                return
            target_units = target_layers // unit_layers
            try:
                source = self.working_file_path
                fmt = 'cif' if source.lower().endswith(('.cif', '.mmcif')) else 'pdb'
                output = os.path.join(os.path.dirname(source), f'temp_modified_{uuid.uuid4().hex[:8]}.{fmt}')
                self.created_temp_files.append(output)
                water_limit = self.get_param(self.edit_water_z, 4.0)
                if target_units < self.detected_units:
                    keep = set()
                    for stack in self.final_sandwiches:
                        count = min(len(stack), target_units)
                        start = (len(stack) - count) // 2
                        keep.update(stack[start:start + count])
                    companions = self.get_kept_heteroatoms(None, 3.0, water_limit, keep_chains=keep)
                    writer = self.write_trimmed_cif if fmt == 'cif' else self.write_trimmed_pdb
                    writer(source, output, keep, companions)
                    message = f'Trimming complete: at most {target_layers} layers per protofilament.'
                else:
                    dialog = self.ExpandDialog(self.detected_units, self)
                    if dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                    writer = self.write_expanded_cif if fmt == 'cif' else self.write_expanded_pdb
                    twist, rise, axis, tilt = writer(source, output, target_units - self.detected_units,
                        water_z_limit=water_limit, **dialog.options)
                    message = (f'Expansion complete: {target_layers} layers per protofilament.\n'
                               f'Twist: {twist:.5f}°; rise per chain unit: {rise:.5f} Å.\n'
                               f'Helical axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]; tilt from Z: {tilt:.2f}°.')
                self._adopt_working_file(output)
                self.operation_log(message)
                self.process_pdb(output)
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Action failed: {exc}")

        def _adopt_working_file(self, path, mapping=None):
            # Carry the parent axis into fragments, including after chain renaming.
            reference = getattr(self, '_axis_reference', None)
            if reference:
                reference['path'] = path
                if mapping:
                    reference['chains'] = {mapping.get(c, c): rows for c, rows in reference['chains'].items()}
            self.working_file_path = path
            audit = path + '.audit.json'
            if os.path.exists(audit) and audit not in self.created_temp_files:
                self.created_temp_files.append(audit)


        def operation_log(self, message):
            log = getattr(self, 'text_area', None) or getattr(self, 'txt', None)
            if log is not None:
                # Explicit markup makes Qt decode escaped text instead of showing entities.
                log.append('<div>' + escape(str(message)).replace('\n', '<br>') + '</div>')


        def write_expanded_structure(self, input_path, output_path, layers_to_add, use_auto,
                                     manual_twist, manual_rise, water_z_limit, use_alt,
                                     use_computed_axis, repeat_units, *, format):
            editor = StructureEditor(input_path)
            result = expand_structure_layers(editor, self.final_sandwiches, layers_to_add,
                use_auto, manual_twist, manual_rise, water_z_limit, use_alt, use_computed_axis,
                label_generator=self.generate_label, repeat_units=repeat_units)
            report = editor.write(output_path, format=format)
            for line in terminal_completion_messages(report):
                self.operation_log(line)
            for warning in report.get('automatic_fit', {}).get('warnings', []):
                self.operation_log('Geometry warning: ' + warning)
            overlap = report.get('overlap_check', {})
            if overlap.get('close_pairs'):
                self.operation_log(f"Geometry warning: {overlap['close_pairs']} C-alpha pairs across placed copies "
                                   f"are closer than {overlap['minimum_new_separation_angstrom']:g} Å. "
                                   'Inspect boundary contacts before using this modeled structure.')
            invalid = [k for k, v in report['metadata'].items() if v['action'] in ('invalidated', 'removed')]
            if invalid:
                self.operation_log('Source metadata invalidated: ' + ', '.join(invalid))
            return result


        def write_expanded_pdb(self, input_path, output_path, layers_to_add, use_auto, manual_twist, manual_rise, water_z_limit=4.0, use_alt=False, use_computed_axis=True, repeat_units=None):
            return self.write_expanded_structure(input_path, output_path, layers_to_add, use_auto,
                manual_twist, manual_rise, water_z_limit, use_alt, use_computed_axis, repeat_units,
                format='pdb')

        def action_restrain(self):
            if not self.final_sandwiches or self.detected_layers == 0: return
            
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Restrain File", "restrain_torsion.cxc", "ChimeraX Command (*.cxc);;All Files (*)")
            if file_path:
                try:
                    mapping = self.create_renaming_mapping(self.final_sandwiches)
                    with open(file_path, 'w') as f:
                        for sandwich in self.final_sandwiches:
                            mid_layer_idx = len(sandwich) // 2
                            mid_chain_original = sandwich[mid_layer_idx]
                            mid_id_new = mapping[mid_chain_original]
                            for layer_k in range(len(sandwich)):
                                if layer_k == mid_layer_idx: continue 
                                target_chain_original = sandwich[layer_k]
                                target_id_new = mapping[target_chain_original]
                                f.write(f"isolde restrain torsions /{target_id_new} template /{mid_id_new} angleRange 180\n")
                    self.text_area.append(f"\n[Export] Restrain file generated: {os.path.basename(file_path)}")
                except Exception as e: QMessageBox.critical(self, "Error", f"Failed to save restrain file:\n{str(e)}")

        def action_connect_chain(self):
            if not self.working_file_path:
                return

            import math
            import shlex

            source_path = self.working_file_path
            temporary_path = source_path + ".connect.tmp"
            target_chain = "A"
            is_cif = source_path.lower().endswith((".cif", ".mmcif"))

            try:
                # Keep every record belonging to a chain together and use the
                # first atom position as the same spatial anchor used by the RF
                # Diffusion merge path.
                chain_records = {}
                first_coordinates = {}

                if is_cif:
                    prefix_lines = []
                    suffix_lines = []
                    atom_headers = []
                    atom_rows = []
                    inside_atom_loop = False
                    atom_loop_seen = False

                    with open(
                        source_path, "r", encoding="utf-8", errors="replace"
                    ) as source:
                        for line in source:
                            stripped = line.strip()

                            if stripped == "loop_":
                                if inside_atom_loop:
                                    inside_atom_loop = False
                                    suffix_lines.append(line)
                                elif atom_loop_seen:
                                    suffix_lines.append(line)
                                else:
                                    prefix_lines.append(line)
                                continue

                            if stripped.startswith("_atom_site."):
                                inside_atom_loop = True
                                atom_loop_seen = True
                                atom_headers.append(stripped.split(".", 1)[1])
                                prefix_lines.append(line)
                                continue

                            if (
                                inside_atom_loop
                                and stripped
                                and not stripped.startswith(("_", "#"))
                            ):
                                try:
                                    fields = shlex.split(
                                        stripped, comments=False, posix=True
                                    )
                                except ValueError:
                                    fields = stripped.split()
                                if len(fields) >= len(atom_headers):
                                    atom_rows.append(fields)
                                continue

                            if inside_atom_loop:
                                inside_atom_loop = False
                            if atom_loop_seen:
                                suffix_lines.append(line)
                            else:
                                prefix_lines.append(line)

                    if not atom_headers or not atom_rows:
                        raise ValueError("No _atom_site records were found.")

                    chain_column = (
                        atom_headers.index("auth_asym_id")
                        if "auth_asym_id" in atom_headers
                        else atom_headers.index("label_asym_id")
                    )
                    x_column = atom_headers.index("Cartn_x")
                    y_column = atom_headers.index("Cartn_y")
                    z_column = atom_headers.index("Cartn_z")
                    sequence_column = (
                        atom_headers.index("auth_seq_id")
                        if "auth_seq_id" in atom_headers
                        else atom_headers.index("label_seq_id")
                    )
                    insertion_column = (
                        atom_headers.index("pdbx_PDB_ins_code")
                        if "pdbx_PDB_ins_code" in atom_headers
                        else None
                    )

                    for fields in atom_rows:
                        chain_id = fields[chain_column]
                        if chain_id not in chain_records:
                            chain_records[chain_id] = []
                            try:
                                first_coordinates[chain_id] = (
                                    float(fields[x_column]),
                                    float(fields[y_column]),
                                    float(fields[z_column]),
                                )
                            except (TypeError, ValueError):
                                first_coordinates[chain_id] = (0.0, 0.0, -9999.0)
                        chain_records[chain_id].append(fields)
                else:
                    header_lines = []
                    footer_lines = []

                    with open(
                        source_path, "r", encoding="utf-8", errors="replace"
                    ) as source:
                        for line in source:
                            record_type = line[:6].strip().upper()
                            if record_type in {"ATOM", "HETATM", "ANISOU"}:
                                if len(line) < 27:
                                    continue
                                chain_id = line[20:22]
                                if chain_id not in chain_records:
                                    chain_records[chain_id] = []
                                    try:
                                        first_coordinates[chain_id] = (
                                            float(line[30:38]),
                                            float(line[38:46]),
                                            float(line[46:54]),
                                        )
                                    except (TypeError, ValueError):
                                        first_coordinates[chain_id] = (
                                            0.0, 0.0, -9999.0
                                        )
                                chain_records[chain_id].append(line)
                            elif record_type == "TER":
                                continue
                            elif (
                                line.startswith("MASTER")
                                or line.startswith("END")
                                or line.startswith("CONECT")
                            ):
                                footer_lines.append(line)
                            else:
                                header_lines.append(line)

                if not chain_records:
                    raise ValueError("No chain records were found.")

                # Match RF Diffusion's geometry-aware ordering. Start at the
                # highest remaining chain, then walk down the closest chain in
                # Z that remains within the same broad XY neighborhood.
                remaining_chains = set(chain_records)
                ordered_chains = []
                while remaining_chains:
                    current_chain = max(
                        remaining_chains,
                        key=lambda chain_id: (
                            first_coordinates[chain_id][2], chain_id
                        ),
                    )
                    ordered_chains.append(current_chain)
                    remaining_chains.remove(current_chain)

                    while True:
                        current_x, current_y, current_z = first_coordinates[
                            current_chain
                        ]
                        candidates = []
                        for candidate in remaining_chains:
                            next_x, next_y, next_z = first_coordinates[candidate]
                            if (
                                current_z - next_z > -2.0
                                and math.hypot(
                                    current_x - next_x, current_y - next_y
                                ) < 25.0
                            ):
                                candidates.append(candidate)

                        if not candidates:
                            break

                        current_chain = min(
                            candidates,
                            key=lambda chain_id: (
                                abs(
                                    current_z
                                    - first_coordinates[chain_id][2]
                                ),
                                -first_coordinates[chain_id][2],
                                chain_id,
                            ),
                        )
                        ordered_chains.append(current_chain)
                        remaining_chains.remove(current_chain)

                # Renumber every source chain into chain A. Two is added at each
                # boundary, so one residue number is intentionally absent; RF
                # Diffusion relies on that gap to preserve the chain break.
                residue_number = 0
                with open(temporary_path, "w", encoding="utf-8") as output:
                    if is_cif:
                        output.writelines(prefix_lines)
                        for chain_id in ordered_chains:
                            residue_number += 2
                            previous_residue = None

                            for fields in chain_records[chain_id]:
                                insertion_code = (
                                    fields[insertion_column]
                                    if insertion_column is not None
                                    else ""
                                )
                                source_residue = (
                                    chain_id,
                                    fields[sequence_column],
                                    insertion_code,
                                )
                                if source_residue != previous_residue:
                                    if previous_residue is not None:
                                        residue_number += 1
                                    previous_residue = source_residue

                                if "auth_asym_id" in atom_headers:
                                    fields[
                                        atom_headers.index("auth_asym_id")
                                    ] = target_chain
                                if "label_asym_id" in atom_headers:
                                    fields[
                                        atom_headers.index("label_asym_id")
                                    ] = target_chain
                                if "auth_seq_id" in atom_headers:
                                    fields[
                                        atom_headers.index("auth_seq_id")
                                    ] = str(residue_number)
                                if "label_seq_id" in atom_headers:
                                    fields[
                                        atom_headers.index("label_seq_id")
                                    ] = str(residue_number)
                                if "label_entity_id" in atom_headers:
                                    fields[
                                        atom_headers.index("label_entity_id")
                                    ] = "1"

                                serialized = [
                                    f"'{field}'"
                                    if " " in field
                                    and not field.startswith(("'", '"'))
                                    else field
                                    for field in fields
                                ]
                                output.write(" ".join(serialized) + "\n")
                        output.writelines(suffix_lines)
                    else:
                        output.writelines(header_lines)
                        for chain_id in ordered_chains:
                            residue_number += 2
                            previous_residue = None

                            for line in chain_records[chain_id]:
                                source_residue = (
                                    chain_id,
                                    line[22:26],
                                    line[26:27],
                                )
                                if source_residue != previous_residue:
                                    if previous_residue is not None:
                                        residue_number += 1
                                    previous_residue = source_residue

                                characters = list(line)
                                if len(characters) < 27:
                                    characters.extend(
                                        " " for _ in range(27 - len(characters))
                                    )
                                characters[20] = " "
                                characters[21] = target_chain
                                characters[22:26] = list(
                                    f"{residue_number:>4}"[-4:]
                                )
                                output.write("".join(characters))
                        output.writelines(footer_lines)

                os.replace(temporary_path, source_path)
                self.text_area.append(
                    "\n>>>> Chains Connected into Chain A "
                    "(RF Diffusion smart top-to-bottom ordering)"
                )
                self.process_pdb(source_path)

            except Exception as error:
                if os.path.exists(temporary_path):
                    try:
                        os.remove(temporary_path)
                    except OSError:
                        pass
                QMessageBox.critical(self, "Error", f"Action failed: {error}")

        def action_save_final(self):
            if not self.working_file_path:
                return
            name, ext = os.path.splitext(self.original_filename_display)
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Final Structure", f"{name}_modified{ext}",
                "Structure Files (*.pdb *.cif *.mmcif);;PDB Files (*.pdb);;CIF Files (*.cif *.mmcif)")
            if not file_path:
                return
            try:
                if not os.path.splitext(file_path)[1]:
                    file_path += '.cif' if selected_filter.startswith('CIF') else ext
                editor = StructureEditor(self.working_file_path)
                editor.rename({})
                editor.write(file_path)
                self.text_area.append(f'Saved validated structure: {file_path}')
                QMessageBox.information(self, "Success", f"Saved: {file_path}\nAudit: {file_path}.audit.json")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Save failed: {exc}")

        # --- NEW HELPER: Detect Spatial & ID Duplicates ---
        def action_evaluate_beta(self):
            if not self.working_file_path: return
            
            import subprocess
            
            # Setup paths (relative execution)
            folder = os.path.dirname(self.working_file_path)
            current_filename = os.path.basename(self.working_file_path)
            # Define naming conventions based on user request
            script_name = f"chimera_script_{current_filename}.py"
            output_filename = f"{current_filename}" 
            
            script_full_path = os.path.join(folder, script_name)
            output_full_path = os.path.join(folder, output_filename)
            
            # 2. Generate Script
            script_content = f"""
import chimera
from chimera import runCommand as rc
rc("open #0 {current_filename}")
rc("ksdssp #0")
rc("write relative #0 #0 {output_filename}")
rc("close session")
"""
            try:
                with open(script_full_path, 'w') as f:
                    f.write(script_content)
                
                # 3. Subprocess
                import platform
                if platform.system() == "Windows":
                    cmd = f'chimera --nogui --script "{script_name}"'
                else:
                    # Use shell=True to handle module loading on HPC clusters
                    cmd = f'module load {CHIMERA_VER} && chimera --nogui --script "{script_name}"'
                
                self.text_area.append(f"\n>>> Re-evaluating secondary structure...")
                # Run inside the folder so relative paths work
                subprocess.run(cmd, shell=True, cwd=folder, check=True)
                
                # 4. Reload
                if os.path.exists(output_full_path):
                    self.created_temp_files.append(output_full_path) # Track new file for cleanup
                    self.working_file_path = output_full_path
                    self.process_pdb(self.working_file_path)
                    self.text_area.append(f">>> β-sheet evaluation completed \n>>> Reloaded PDB file")
                else:
                    self.text_area.append("Error: Output file not found after Chimera run.")
                    
            except Exception as e:
                self.text_area.append(f"\n>>> Chimera not found or failed. Skipping β-sheet evaluation. (Error: {e})")
            finally:
                # Always clean up the temporary script
                if os.path.exists(script_full_path):
                    try: os.remove(script_full_path)
                    except: pass

        def check_spatial_and_id_duplicates(self, file_path):
            # Reject duplicate identities rather than leave ambiguous metadata references.
            StructureEditor(file_path)

        def rewrite_split_chains(self, file_path, split_indices):
            raise StructureEditError('Ambiguous chain splitting requires an explicit residue mapping; no automatic renaming was performed.')

        # ================= LOGIC: ANALYSIS (Based on Layer_identifyer.py) =================
        def process_pdb(self, filename, old_sandwiches=None, rename_map=None):
            try:
                editor = StructureEditor(filename)
                chains, waters, ions = editor.preview()
                for notice in new_source_notices(self, editor.report):
                    self.operation_log('Warning: ' + notice)
                hint = None
                reference = getattr(self, '_axis_reference', None)
                if reference and reference['path'] == filename:
                    p, q = [], []
                    for cid, rows in chains.items():
                        old = reference['chains'].get(cid, {})
                        for row in rows:
                            if row['id'] in old:
                                p.append(old[row['id']]); q.append(row['coord'])
                    if len(p) >= 6:
                        fit = _fit_detection_core(np.asarray(p), np.asarray(q))
                        if fit is not None and fit['rmsd'] <= 1.0:
                            hint = fit['rotation'] @ np.asarray(reference['axis'])
                result = detect_layers(chains, axis_hint=hint)
                if hint is not None and reference.get('antiparallel', False) and result['orientation'] == 'undetermined':
                    result['parent_antiparallel'] = True
                    result['warnings'].append('The parent stack was antiparallel; this fragment has insufficient neighbors to recheck its orientation.')
                pattern = _expansion_repeat_pattern(chains, result['sandwiches'])
                result['suggested_repeat_units'] = pattern['period']
                result['compatible_repeat_units'] = [p for p in range(1, result['detected_units'] + 1)
                                                     if pattern['compatible'](p)]
                if result.get('parent_antiparallel'):
                    result['suggested_repeat_units'] = max(2, result['suggested_repeat_units'])
                    result['compatible_repeat_units'] = [p for p in result['compatible_repeat_units'] if p % 2 == 0]
                self.detection_result = result
                self.update_water_axial_limit()
                self.final_sandwiches = result['sandwiches']
                self.detected_units = result['detected_units']
                self.detected_layers = result['detected_layers']
                self.layers_per_unit = result['layers_per_unit']
                if result['axis'] is not None:
                    self._axis_reference = dict(path=filename, axis=result['axis'],
                        antiparallel=result['antiparallel'] or result.get('parent_antiparallel', False),
                        chains={c: {r['id']: r['coord'] for r in rows} for c, rows in chains.items()})
                self.results = {c: {'centroid': self.get_centroid(rows)} for c, rows in chains.items()}
                self.chains_data_plot = {c: [tuple(r['coord']) for r in rows] for c, rows in chains.items()}
                self.current_waters, self.current_ions = waters, ions
                self.lbl_orientation.setText(result['orientation'].capitalize())
                lines = [f'--------------- Amyloid Modifier {VERSION} Detection ---------------',
                         result['orientation'].capitalize(),
                         f'Total Chains: {len(chains)}',
                         f'Protofilaments: {len(self.final_sandwiches)}',
                         f'Max Chain Units: {self.detected_units}',
                         f'Max Physical Layers: {self.detected_layers}']
                if result['axis'] is None:
                    lines.append('Stacking axis: undetermined (no inter-unit repeat)')
                else:
                    lines.append('Stacking axis (' + result['axis_source'] + '): [' + ', '.join(f'{v:.4f}' for v in result['axis']) + ']')
                for i, track in enumerate(result['protofilaments']):
                    lines.append(f"\nProtofilament #{i+1} ({track['orientation']}): {track['units']} units × {track['layers_per_unit']} layers/unit = {track['layers']} layers")
                    period = result['suggested_repeat_units']
                    if track['orientation'] == 'antiparallel' and track['complete'] and period > 1:
                        repeats, remainder = divmod(track['units'], period)
                        summary = f"Repeat units: {repeats} complete × {period} chains/repeat"
                        if remainder:
                            summary += f" + {remainder} unpaired chain units"
                        lines.append(summary)
                    lines.append(', '.join(track['chains']))
                    if not track['complete']:
                        lines.append(f"Unit positions: {track['unit_positions']}; axial span: {track['axial_layer_span']} layers")
                if old_sandwiches and rename_map:
                    lines.append('\nRenamed chains: ' + ', '.join(f'{c} → {rename_map.get(c, c)}' for s in old_sandwiches for c in s))
                lines.extend('Note: ' + w for w in result['warnings'])
                self.operation_log('\n'.join(lines))
                self.grp_ops.setEnabled(True)
                self.btn_save.setEnabled(True)
                self.btn_beta.setEnabled(self.detected_layers > 1)
                self.cmb_protofilaments.blockSignals(True)
                self.cmb_protofilaments.clear()
                for target in protofilament_selection_targets(editor, result, self.get_param(self.edit_water_z, 4.0)):
                    self.cmb_protofilaments.addItem(target['label'], userData=target)
                    self.cmb_protofilaments.setItemData(self.cmb_protofilaments.count() - 1,
                        target['label'], Qt.ItemDataRole.ToolTipRole)
                self.cmb_protofilaments.blockSignals(False)
                highlights = {s[len(s)//2] for s in self.final_sandwiches}
                self.mol_canvas.plot_chains(self.chains_data_plot, label_ids=highlights, waters=waters, ions=ions)
            except Exception as exc:
                # A failed new analysis must not leave a previous model's assignment active.
                self.detection_result = None
                self.final_sandwiches = []
                self.results = {}
                self.detected_units = self.detected_layers = 0
                self.layers_per_unit = 1
                self._axis_reference = None
                self.lbl_orientation.setText('Undetermined')
                self.cmb_protofilaments.clear()
                self.grp_ops.setEnabled(False)
                self.btn_beta.setEnabled(False)
                self.update_water_axial_limit()
                self.operation_log(f'Analysis Error: {exc}')

        def parse_cif_manual_logic(self, filepath):
            return StructureEditor(filepath).preview()

        def parse_pdb_manual_logic(self, filepath):
            return StructureEditor(filepath).preview()

        def get_centroid(self, chain_data):
            if not chain_data: return np.array([0.0, 0.0, 0.0])
            coords = [res['coord'] for res in chain_data]
            return np.mean(coords, axis=0)


        def get_expansion_order(self, n_layers):
            mid = n_layers // 2; order = [mid]; offset = 1
            while True:
                added = False
                up = mid + offset
                if up < n_layers: order.append(up); added = True
                down = mid - offset
                if down >= 0: order.append(down); added = True
                if not added: break
                offset += 1
            return order

        def generate_label(self, index):
            if self.chk_only_numbers.isChecked(): return str(index)
            use_numbers = self.chk_numbers_first.isChecked()
            if use_numbers:
                if index < 26: return chr(ord('A') + index)
                elif index < 36: return str(index - 26)
                else:
                    temp_index = index - 10; label = ""; temp_index += 1
                    while temp_index > 0:
                        temp_index -= 1; label = chr(ord('A') + (temp_index % 26)) + label; temp_index //= 26
                    return label
            else:
                label = ""; index += 1
                while index > 0: 
                    index -= 1; label = chr(ord('A') + (index % 26)) + label; index //= 26
                return label

        def create_renaming_mapping(self, sandwiches):
            editor = StructureEditor(self.working_file_path)
            core = {c for s in sandwiches for c in s}
            reserved = set(editor.chains) - core
            mapping, index = {}, 0
            for layer in self.get_expansion_order(self.detected_units):
                for sandwich in sandwiches:
                    if layer >= len(sandwich):
                        continue
                    while True:
                        label = self.generate_label(index)
                        index += 1
                        if label not in reserved and label not in mapping.values():
                            break
                    mapping[sandwich[layer]] = label
            return mapping
        
        def write_renamed_cif(self, input_filename, output_filename, sandwiches):
            editor = StructureEditor(input_filename)
            editor.rename(self.create_renaming_mapping(sandwiches))
            editor.write(output_filename, format='cif')

        def write_trimmed_cif(self, input_path, output_path, keep_chains, keep_het_lines):
            editor = StructureEditor(input_path)
            # Selections are complete residue identities, never physical text line numbers.
            residues = keep_het_lines or set()
            editor.trim(keep_chains, keep_residues=residues, polymer_only=True)
            report = editor.write(output_path, format='cif')
            self.text_area.append('Validated trim: {} atoms.'.format(report['atoms']))

        def write_expanded_cif(self, input_path, output_path, layers_to_add, use_auto, manual_twist, manual_rise, water_z_limit=4.0, use_alt=False, use_computed_axis=True, repeat_units=None):
            return self.write_expanded_structure(input_path, output_path, layers_to_add, use_auto,
                manual_twist, manual_rise, water_z_limit, use_alt, use_computed_axis, repeat_units,
                format='cif')

        def write_renamed_pdb(self, input_filename, output_filename, sandwiches):
            editor = StructureEditor(input_filename)
            editor.rename(self.create_renaming_mapping(sandwiches))
            editor.write(output_filename, format='pdb')

        def write_trimmed_pdb(self, input_path, output_path, keep_chains, keep_atom_indices=None):
            editor = StructureEditor(input_path)
            # Selections are complete residue identities, never physical text line numbers.
            residues = keep_atom_indices or set()
            editor.trim(keep_chains, keep_residues=residues, polymer_only=True)
            report = editor.write(output_path, format='pdb')
            self.text_area.append('Validated trim: {} atoms.'.format(report['atoms']))



    class ModifierLayerViewerTabs(QTabWidget):
        def __init__(self):
            super().__init__()
            self.pdb_widget = PDBLayerIdentifier()
            self.cif_widget = CIFLayerIdentifier()
            self.addTab(self.pdb_widget, "Modifier")
            self.addTab(self.cif_widget, "Layer Viewer")

    # Tests and embedding callers can obtain the exact components without
    # creating windows or entering QApplication.exec().
    if not launch:
        return {
            "OpenGLChainCanvas": OpenGLChainCanvas,
            "SoftwareChainCanvas": SoftwareChainCanvas,
            "ChainCanvas": ChainCanvasBase,
            "CIFLayerIdentifier": CIFLayerIdentifier,
            "PDBLayerIdentifier": PDBLayerIdentifier,
            "ModifierLayerViewerTabs": ModifierLayerViewerTabs
        }

    qt_app = QApplication.instance()
    if not qt_app:
        qt_app = QApplication(sys.argv)
    
    qt_app.setStyleSheet("""
        QMainWindow, QDialog, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Arial; font-size: 8pt; }
        
        QPushButton { background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42; padding: 5px 15px; border-radius: 4px; }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed, QPushButton:checked { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }

        QPushButton#submitValidationButton,
        QPushButton#runExtractionButton,
        QPushButton#iterateDesignsButton {
            min-height: 28px;
            padding: 5px 12px;
            font-weight: normal;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QTextBrowser, QTableWidget, QTableView, QComboBox { 
            background-color: #1e1e1e; border: 1px solid #3e3e42; color: #cccccc; padding: 2px; border-radius: 2px;
            selection-background-color: #98c379; selection-color: #1e1e1e;
        }
        QTableView { alternate-background-color: #252526; gridline-color: #3e3e42; }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 20px; 
        }

        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { 
            background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; outline: none;
            selection-background-color: #98c379; selection-color: #1e1e1e;
        }
        QComboBox QAbstractItemView::item { padding: 2px 5px; min-height: 16px; border: none !important; }
        QComboBox QAbstractItemView::item:hover,
        QComboBox QAbstractItemView::item:selected { 
            background-color: #98c379; color: #1e1e1e; border: none !important; outline: none !important;
        }

        QTreeWidget, QListWidget, QTreeView { 
            background-color: #252526; border: 1px solid #3e3e42; color: #cccccc; outline: none; 
        }
        QTreeWidget::item, QListWidget::item { padding: 5px; }
        QTreeWidget::item:selected, QListWidget::item:selected { 
            background-color: #3e3e42; color: #ffffff; border-left: 3px solid #98c379; 
        }

        QHeaderView::section { background-color: #252526; color: #98c379; border: 1px solid #3e3e42; padding: 4px; font-weight: bold; }
        QTableCornerButton::section { background-color: #252526; border: 1px solid #3e3e42; }

        QMenu { background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; }
        QMenu::item:selected { background-color: #3e3e42; color: #98c379; }

        QProgressBar { 
            border: 1px solid #3e3e42; border-radius: 2px; 
            background-color: #1e1e1e; text-align: center; color: #d4d4d4; 
        }
        QProgressBar::chunk { background-color: #98c379; border-radius: 2px; color: #1e1e1e; }

        QGroupBox { border: 1px solid #3e3e42; border-radius: 4px; margin-top: 1.0em; font-weight: bold; color: #98c379; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 2px; background-color: #1e1e1e; }
        QCheckBox::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        QRadioButton::indicator { width: 12px; height: 12px; border: 1px solid #555; border-radius: 7px; background-color: #1e1e1e; }
        QRadioButton::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        
        QSlider:vertical { min-width: 20px; }
        QSlider::groove:vertical { background: #3c3c3c; width: 6px; border-radius: 3px; }
        QSlider::handle:vertical { background: #98c379; height: 14px; width: 14px; margin: 0 -4px; border-radius: 7px; }
        QSlider::handle:vertical:hover { background: #b5e890; }
        
        QSlider:horizontal { min-height: 20px; }
        QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #98c379; height: 14px; width: 14px; margin: -4px 0; border-radius: 7px; }
        QSlider::handle:horizontal:hover { background: #b5e890; }
        
        QTabWidget::tab-bar { alignment: left; }
        QTabWidget::pane { border: 1px solid #3e3e42; top: -1px; }
        QTabBar::tab { background: #252526; border: 1px solid #3e3e42; padding: 6px 12px; min-width: 100px; color: #d4d4d4; }
        QTabBar::tab:selected { background: #3e3e42; color: #98c379; font-weight: bold; border-bottom: 1px solid #3e3e42; }
        QTabBar::tab:hover { background: #4e4e52; }
        
        QScrollBar:vertical { background: transparent; width: 12px; margin: 0px; }
        QScrollBar::handle:vertical { background-color: #4e4e52; min-height: 20px; border-radius: 5px; margin: 2px; }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed { background-color: #98c379; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; background: transparent; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        
        QScrollBar:horizontal { background: transparent; height: 12px; margin: 0px; }
        QScrollBar::handle:horizontal { background-color: #4e4e52; min-width: 20px; border-radius: 5px; margin: 2px; }
        QScrollBar::handle:horizontal:hover, QScrollBar::handle:horizontal:pressed { background-color: #98c379; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; background: transparent; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
    """)


    win = ModifierLayerViewerTabs()
    renderer_suffix = " (Software Renderer)" if use_software_renderer else ""
    win.setWindowTitle(f"Modifier {VERSION} + Layer Viewer{renderer_suffix}")
    win.resize(1250, 800)
    win.show()

    sys.exit(qt_app.exec())

if __name__ == "__main__":
    open_modifier_layer_viewer()
