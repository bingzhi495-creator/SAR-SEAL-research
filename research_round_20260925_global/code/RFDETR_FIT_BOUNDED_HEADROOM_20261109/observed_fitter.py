"""Readout adaptation of the P143 bounded original-raster teacher fitter.

The solver, support, native recheck, and acceptance predicates are carried forward.
The only added behavior records the four original-grid alternative masks.
"""
from _p150_teacher_fit import Counter, SEED, np, render, sample_points, sha256


def build_image(env,item,solver,readout):
    t=env.torch
    c=env.image(item)
    anns,truth,matches,rles=env.matches(item,c)
    readout.start(c,truth,matches,rles)
    z,q,f,ctx,hw=(c[k] for k in ("z","q","basis","ctx","hw"))
    pairs=[(j,k) for j,k in matches if bool(ctx["active"][c["query"][j]])]
    totals=Counter(matched=len(matches),active_matched=len(pairs),suppressed_support=len(matches)-len(pairs))
    if not pairs:
        return [],totals
    staged=[]
    f_flat=f.flatten(1).T.detach().cpu().numpy().astype(np.float64)
    global_gram=f_flat.T@f_flat/len(f_flat)
    proposed=t.zeros_like(q)
    offsets=t.zeros((len(q),1,1),dtype=z.dtype,device=z.device)
    for pos,k in pairs:
        qi=c["query"][pos]
        native=c["native"][pos]
        gt=truth[k]
        stock=env.geometry.signed_counts(native,gt)
        support=env.geometry.sampling_support(gt,native,SEED+item["dataset_index"]*100+qi)
        idx=support["index"]
        label=gt.ravel()[idx]
        weights=support["weights"]
        totals.update(already_strict=stock["strict"],fit_points=len(idx),small=int(stock["area"]<1024))
        kept_held=bool((gt & support["held"]).any())
        delta=np.zeros(256,np.float32)
        if not stock["strict"]:
            local=ctx["local"][qi].flatten().cpu().numpy()
            q0=q[qi].detach().cpu().numpy().astype(np.float64)
            local_f=f_flat[local]
            radius=float(ctx["radius"][qi])
            metric=(global_gram+local_f.T@local_f/len(local_f))*(.5/radius**2)+np.eye(256)/max(q0@q0,1e-12)
            design=sample_points(f,idx,hw,t).cpu().numpy().astype(np.float64)
            base=sample_points(z[qi:qi+1],idx,hw,t)[:,0].cpu().numpy().astype(np.float64)
            fitted=solver.solve_bounded(design,metric,base,label.astype(np.float64),weights)
            scalar=solver.solve_bounded(np.ones((len(idx),1)),np.array([[1/radius**2]]),base,
                                        label.astype(np.float64),weights)
            delta=fitted.iterative.astype(np.float32)
            proposed[qi]=t.from_numpy(delta).to(device=q.device)
            offsets[qi]=float(scalar.iterative[0])
            totals.update(non_strict=1,small_non_strict=int(stock["area"]<1024),
                          projection_stall=int(fitted.projected_gradient>.01))
        staged.append((pos,k,qi,stock,kept_held,support,delta))
    with t.no_grad():
        coeff=env.head.bounded_readout(f,q,z,ctx,proposed)
        queries=[x[2] for x in staged]
        generated=[render(planes[queries],hw).gt(0).cpu().numpy() for planes in (z,coeff,z+offsets)]
    rows=[]
    for i,(pos,k,qi,stock,held_gt,support,delta) in enumerate(staged):
        original,cf,off=[v[i] for v in generated]
        gt=truth[k]
        env.s.need(np.array_equal(original,c["native"][pos]),"Exact original stock mask sign")
        held=support["held"]
        raw=[]
        for name,v in (("coeff",cf),("offset",off)):
            m=env.geometry.signed_counts(v,gt)
            a=env.geometry.signed_counts(v[held],gt[held])["error"]-env.geometry.signed_counts(
                    original[held],gt[held])["error"]
            keep=held_gt and env.geometry.accepted(stock,m,a,stock["area"])
            raw.append((keep,m))
            if not stock["strict"]:
                totals.update({f"{name}_raw_crossings":m["strict"],f"{name}_raw_fp_delta":m["fp"]-stock["fp"],
                    f"{name}_raw_fn_delta":m["fn"]-stock["fn"],f"{name}_accepted":int(keep),
                    f"{name}_verified_crossings":int(keep)*m["strict"],
                    f"{name}_accepted_small":int(keep and stock["area"]<1024),
                    f"{name}_verified_crossings_small":int(keep and stock["area"]<1024)*m["strict"]})
        readout.replace(pos,stock,gt,raw,cf,off)
        keep,_=raw[0]
        idx=support["index"]
        rows.append(dict(query=qi,native_rank=c["rank"][pos],annotation_id=anns[k]["id"],
                         stock_mask_sha256=sha256(original.tobytes()).hexdigest(),keep=bool(keep),
                         strict_before=stock["strict"],index=t.from_numpy(idx),labels=t.from_numpy(gt.ravel()[idx]),
                         weights=t.from_numpy(support["weights"].astype(np.float32)),
                         delta=t.from_numpy(delta if keep else np.zeros_like(delta))))
    return rows,totals

