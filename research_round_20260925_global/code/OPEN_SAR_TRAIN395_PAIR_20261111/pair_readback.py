"""Numerical and annotation-frame readback for the fixed OpenSAR TRAIN 395 VH pair."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

NAMES = {
    "grd": ("DB_OPENSAR_VD_395_GRD_VH.tiff", "f4b89b10629782507f08ac129ba34cba60884b889947bc05ddc0894e2b65b770"),
    "slc": ("DB_OPENSAR_VD_395_SLC_VH.tiff", "cec09dce323a64d97e70ce647d0abed65e77bbccc1f6f7bea113feed7078f072"),
    "xml": ("DB_OPENSAR_VD_395.xml", "9457786b4d7a15aa40807453d3d802b641c6dc16d5e6334175185f9faedc06ce"),
    "rc": ("s1a-iw-raw-s-vh-20200407t052956-20200407t053029-032017-03b2e2-IW2-VD_395_rescaled.npy",
           "0f7c79f3c800b44d2fdebf84be6e7a6f032274c28c36bb46884858defe7a1b2a"),
}
GEO_TAGS = (33550, 33922, 34264, 34735)


def read_one(root, key):
    name, digest = NAMES[key]
    path = root / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 4_000_000:
        raise ValueError("exact bounded selected member required")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError("selected member bytes changed")
    return raw


def read_tiff(raw):
    with tifffile.TiffFile(BytesIO(raw)) as tf:
        if len(tf.pages) != 1:
            raise ValueError("one band required")
        page = tf.pages[0]
        return page.asarray(), any(tag in page.tags for tag in GEO_TAGS)


def box(ship, name):
    return tuple(int(ship.findtext(f"{name}/{key}")) for key in ("Top","Left","Bottom","Right"))


def peak_and_flanks(profile, x1, x2):
    width = x2-x1
    flanks = np.r_[profile[x1-width:x1], profile[x2:x2+width]]
    return dict(global_peak_col0=int(profile.argmax()), box_peak_col0=int(x1+profile[x1:x2].argmax()),
                mean_box_to_flanks_db=float(10*np.log10(profile[x1:x2].mean()/flanks.mean())))


def run(ground, focused):
    root = ET.fromstring(read_one(ground,"xml"))
    ships = root.findall("./ProcessingData/List_of_ships/Ship")
    if len(ships)!=1 or ships[0].findtext("Is_Vessel")!="True" or ships[0].findtext("Confidence")!="HIGH":
        raise ValueError("pinned one-ship positive label required")
    ship = ships[0]
    corner = root.find("./ProcessingData/Corner_Coord")
    x0 = min(map(int,corner.findtext("SARData_Sample").split()))
    y0 = min(map(int,corner.findtext("SARData_Line").split()))
    t,l,b,r = box(ship,"BoundingBox")
    rt,rl,rb,rr = box(ship,"BoundingBox_RC_Local")
    gt,gl,gb,gr = box(ship,"BoundingBox_RC_Global")
    cx = int(ship.findtext("Centroid_Position/Scene_Sample"))
    cy = int(ship.findtext("Centroid_Position/Scene_Line"))
    if (x0+l-1,x0+r-1)!=(gl,gr) or (gl-rl,gr-rr)!=(x0,x0) or (gt-rt)!=(gb-rb):
        raise ValueError("published range and RC azimuth origins inconsistent")
    if (x0+cx-1,y0+cy-1)!=(int(ship.findtext("Centroid_Position/SARData_Sample")),
                           int(ship.findtext("Centroid_Position/SARData_Line"))):
        raise ValueError("published SLC centroid origins inconsistent")
    grd,ggeo = read_tiff(read_one(ground,"grd"))
    slc,sgeo = read_tiff(read_one(focused,"slc"))
    rc = np.load(BytesIO(read_one(ground,"rc")), allow_pickle=False)
    if grd.shape!=(708,180) or grd.dtype!=np.uint16 or slc.shape!=(512,512) or rc.shape!=(691,512):
        raise ValueError("selected array shapes differ")
    if slc.dtype!=np.complex64 or rc.dtype!=np.complex64 or not all(np.isfinite(x).all() for x in (slc,rc)):
        raise ValueError("both selected IQ arrays must be finite complex64")
    if not (1<=t<=b<=slc.shape[0] and 1<=l<=r<=slc.shape[1] and 0<=rt<=rb<rc.shape[0]
            and 0<=rl<=rr<rc.shape[1] and (l-1,r)==(rl,rr+1)):
        raise ValueError("fixed native boxes inconsistent")
    slc_profile = (np.abs(slc[t-1:b].astype(np.complex128))**2).mean(axis=0)
    rc_profile = (np.abs(rc[rt:rb+1].astype(np.complex128))**2).mean(axis=0)
    lo,hi = rl-16,rr+17
    profile_corr = float(np.corrcoef(np.log(slc_profile[lo:hi]),np.log(rc_profile[lo:hi]))[0,1])
    source = rc.astype(np.complex128)
    rd = np.fft.fft(source,axis=0,norm="ortho")
    restored = np.fft.ifft(rd,axis=0,norm="ortho")
    axis = np.arange(rc.shape[0])
    column = source[:,cx-1]
    dft = np.exp(-2j*np.pi*(axis[:,None]*axis[None,:])/len(axis))@column/np.sqrt(len(axis))
    return dict(status="OPEN_SAR_TRAIN395_NATIVE_PAIR_NUMERIC_READBACK", train_only=True,
                arrays={k:dict(shape=list(x.shape),dtype=str(x.dtype),sha256=NAMES[k][1])
                        for k,x in (("grd",grd),("slc",slc),("rc",rc))},
                annotation=dict(slc_box_tlbr=[t,l,b,r], rc_local_tlbr=[rt,rl,rb,rr],
                                slc_range_origin=x0,slc_azimuth_origin=y0,rc_azimuth_origin=gt-rt,
                                common_global_range_box=[gl,gr],same_local_range_columns0=[rl,rr],
                                grd_geotags_present=ggeo,slc_geotags_present=sgeo,
                                slc_box_fits_grd_native_shape=bool(b<=grd.shape[0] and r<=grd.shape[1])),
                native_profiles=dict(slc=peak_and_flanks(slc_profile,rl,rr+1),
                                     rc=peak_and_flanks(rc_profile,rl,rr+1),
                                     log_power_pearson_box_plus_16_columns=profile_corr),
                rd_numeric=dict(fft_axis=0,prf_calibrated=False,frequency_spacing_cycles_per_input_row=1/len(axis),
                                relative_fft_ifft_l2=float(np.linalg.norm(restored-source)/np.linalg.norm(source)),
                                direct_dft_vs_fft_col0=cx-1,
                                relative_direct_dft_l2=float(np.linalg.norm(dft-rd[:,cx-1])/np.linalg.norm(dft))),
                scene_registration_independently_confirmed=False, detection_evaluated=False, mask_evaluated=False)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for option in ("ground","focused","output"):
        cli.add_argument("--"+option,type=Path,required=True)
    a=cli.parse_args()
    if a.output.exists() or not a.output.parent.is_dir():
        raise ValueError("fresh result required")
    result = run(a.ground,a.focused)
    with a.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
