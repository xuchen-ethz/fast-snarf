# Rigging Static Scans using Fast-SNARF

<img src="assets/anim.gif" width="250" height="250"/> 

Here is a quick guide to rig a static scan using Fast-SNARF. This guide assumes you have already installed Fast-SNARF dependencies following [here](../../README.md). 

Given a single scan, this small project 
 - turns the scan into a rigged character that can be animated,
 - takes around a minute,
 - can handle loose clothing, such as skirts
This is done via canonicalizing the scan into a rest pose using forward skinning and the SMPL skinning weights diffused in the canonical volume.

Run the script in the main folder of Fast-SNARF via
```
python projects/rig_scan/run.py \
    --obj_path <path to .obj> 
    --param_path <path to .pkl>
    --tex_path <path to .png>
    [--with_tex]
    [--loose_cloth]
```

The parameters are:

`obj_path`: path to the .obj file of the scan

`param_path`: path to the registered smpl parameters of the scan

`tex_path`: path to the texture of the scan

`with_tex`: if specified, will learn an animataibel textured scan

`loose_cloth`: set to true if handling loose clothing like skirts


