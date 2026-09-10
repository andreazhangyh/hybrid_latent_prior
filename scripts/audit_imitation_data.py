"""Offline provenance and physical-scale audit of original/corrected datasets."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'isaacgymenvs/tasks/amp/poselib'))
from poselib.skeleton.skeleton3d import SkeletonMotion


def main():
    base = ROOT / 'isaacgymenvs/tasks/amp/poselib/data/AMP'
    limits = {}
    xml = ET.parse(str(ROOT / 'assets/mjcf/amp_humanoid.xml'))
    for body in xml.findall('.//body'):
        joints = body.findall('joint')
        if joints:
            limits[body.attrib['name']] = np.deg2rad([list(map(float, j.attrib['range'].split())) for j in joints])
    report = {}
    for label, directory in [('original', base/'LAFAN_ALL_2026-Sep-07'),
                             ('corrected_v2', base/'LAFAN_ALL_corrected_v2_2026-Sep-09')]:
        rows = []
        for path in sorted(directory.rglob('*.npy')):
            m = SkeletonMotion.from_file(str(path))
            all_excess = []
            for name, bounds in limits.items():
                q = m.local_rotation[:, m.skeleton_tree.index(name)].numpy()
                rotvec = Rotation.from_quat(q).as_rotvec()
                dof = rotvec if len(bounds) == 3 else rotvec[:, 1:2]
                all_excess.append(np.maximum(bounds[:, 0] - dof, dof - bounds[:, 1]).clip(0))
            excess = np.concatenate(all_excess, -1)
            speed = m.global_root_velocity.norm(dim=-1).numpy()
            heights = m.root_translation[:, 2].numpy()
            rows.append(dict(path=str(path.relative_to(directory)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=len(heights),
                root_height_m=[float(x) for x in np.percentile(heights,[0,50,100])],
                root_speed_m_s=[float(x) for x in np.percentile(speed,[50,95,100])],
                joint_limit_violation_fraction=float((excess.max(-1) > .05).mean()),
                max_joint_limit_excess_rad=float(excess.max())))
        report[label] = rows
        total=sum(x['frames'] for x in rows)
        print(label, 'files',len(rows),'frames',total,
              'limit violation fraction',sum(x['frames']*x['joint_limit_violation_fraction'] for x in rows)/total)
        print('walk1',next(x for x in rows if x['path']=='train/walk1_subject1.npy'))
    dest=ROOT/'artifacts/imitation_evaluation/data_audit.json'
    dest.write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
