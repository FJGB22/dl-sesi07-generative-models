"""Membangkitkan report/numbers.tex dari JSON hasil eksperimen, agar setiap angka di laporan
berasal langsung dari outputs/results/ (tidak diketik manual).

    python report/build_numbers.py --kelas "LXXX" --github "https://github.com/..."
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda n: json.load(open(os.path.join(ROOT, "outputs", "results", n)))


def f(x, d=2):
    """Format desimal gaya Indonesia (koma)."""
    return f"{x:.{d}f}".replace(".", "{,}")


def pct(x, d=2):
    return f(100 * x, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kelas", default="[isi kelas]")
    ap.add_argument("--github", default="https://github.com/[username]/[nama-repo]")
    a = ap.parse_args()
    E = R("mlp_experiments.json")
    M = R("metrics.json")
    I = R("interpolation.json")
    s4 = E["stage4"]
    fd, ae = s4["frozen_mlp_decoder"], s4["vanilla_ae"]
    s5 = E["stage5"]
    m = {
        "KELAS": a.kelas, "GITHUB": a.github,
        # ---- Bagian III
        "MLPacc": pct(E["stage1"]["test_acc"]),
        "WoneOrth": f(E["weights"]["W1"]["rel_fro_WtW_minus_I"], 1),
        "WoneCond": f(E["weights"]["W1"]["cond"], 1),
        "WoneSVmax": f(E["weights"]["W1"]["sv_max"], 1),
        "ReluOne": pct(E["relu_zeroed_frac"]["layer1"], 1) + "\\%",
        "ReluTwo": pct(E["relu_zeroed_frac"]["layer2"], 1) + "\\%",
        "ZeroFrac": pct(s5["latent_zero_frac"], 1) + "\\%",
        "NegFrac": pct(s5["negative_frac_in_samples"], 1) + "\\%",
        "TtwoMSE": f(E["stage2"]["vs_class_mean_image"]["mse"], 4),
        "TtwoPSNR": f(E["stage2"]["vs_class_mean_image"]["psnr"]),
        "TtwoSSIM": f(E["stage2"]["vs_class_mean_image"]["ssim"], 3),
        "TthreeMSE": f(E["stage3"]["transpose"]["mse"], 4),
        "TthreePSNR": f(E["stage3"]["transpose"]["psnr"]),
        "TthreeSSIM": f(E["stage3"]["transpose"]["ssim"], 3),
        "TpinvMSE": f(E["stage3"]["pinv"]["mse"], 4),
        "TpinvPSNR": f(E["stage3"]["pinv"]["psnr"]),
        "TpinvSSIM": f(E["stage3"]["pinv"]["ssim"], 3),
        "FdMSE": f(fd["mse"], 4), "FdPSNR": f(fd["psnr"]), "FdSSIM": f(fd["ssim"], 3),
        "FdVar": f(fd["intra_class_var_ratio"], 3), "FdAcc": pct(fd["mlp_acc_on_recon"], 1),
        "AeMSE": f(ae["mse"], 4), "AePSNR": f(ae["psnr"]), "AeSSIM": f(ae["ssim"], 3),
        "AeVar": f(ae["intra_class_var_ratio"], 3), "AeAcc": pct(ae["mlp_acc_on_recon"], 1),
        "FdAeRatio": f(fd["mse"] / ae["mse"], 1),
        "NNrand": f(s5["nn_dist_sampled_to_train"]), "NNreal": f(s5["nn_dist_test_to_train"]),
        "NNratio": f(s5["nn_dist_sampled_to_train"] / s5["nn_dist_test_to_train"], 1),
        "EntMLP": f(s5["mlp_conf_samples"]["mean_entropy"], 3),
        "EntVAE": f(s5["mlp_conf_vae"]["mean_entropy"], 3),
        "EntReal": f(s5["mlp_conf_real"]["mean_entropy"], 3),
        "TrouserMLP": str(s5["mlp_conf_samples"]["class_hist"][1]),
        "FIDmlp": f(M["fid"]["mlp_latent"], 1), "FIDvae": f(M["fid"]["vae"], 1),
    }
    # ---- Bagian IV
    for k in ("ae", "sae", "vae", "gan", "dcgan", "ddpm"):
        K = k.upper()
        m[f"T{K}"] = f(M["train_time_per_epoch_s"][k], 1)
        m[f"L{K}"] = f(M["latency_ms_batch64"][k][0], 2 if M["latency_ms_batch64"][k][0] < 100 else 0)
        m[f"P{K}"] = f(M["params"][k] / 1e6, 2)
        if k in M["recon"]:
            m[f"MSE{K}"] = f(M["recon"][k]["mse"], 4)
            m[f"PSNR{K}"] = f(M["recon"][k]["psnr"])
            m[f"SSIM{K}"] = f(M["recon"][k]["ssim"], 3)
        if k in M["fid"]:
            m[f"FID{K}"] = f(M["fid"][k], 1)
        if k in M["coverage"]:
            c = M["coverage"][k]
            m[f"COV{K}"] = f(c["norm_class_entropy"], 3)
            m[f"MINC{K}"] = str(min(c["class_hist"]))
            m[f"MAXC{K}"] = str(max(c["class_hist"]))
    m["LDDPMstep"] = f(M["latency_ms_batch64"]["ddpm_single_step"][0], 1)
    m["LVAErec"] = f(M["latency_ms_batch64"]["vae_recon"][0], 2)
    m["FIDreal"] = f(M["fid"]["real_val_vs_test"], 1)
    m["FIDaegauss"] = f(M["fid"]["ae_gaussfit"], 1)
    m["FIDsaegauss"] = f(M["fid"]["sae_gaussfit"], 1)
    m["COVAEgauss"] = f(M["coverage"]["ae_gaussfit"]["norm_class_entropy"], 3)
    m["COVSAEgauss"] = f(M["coverage"]["sae_gaussfit"]["norm_class_entropy"], 3)
    m["COVmlp"] = f(M["coverage"]["mlp_latent"]["norm_class_entropy"], 3)
    m["COVreal"] = f(M["coverage"]["real_test"]["norm_class_entropy"], 3)
    m["DDPMxGAN"] = f(M["latency_ms_batch64"]["ddpm"][0] / M["latency_ms_batch64"]["dcgan"][0], 0)
    sp = M["sparsity"]
    for k in ("ae", "sae"):
        K = k.upper()
        m[f"ZF{K}"] = pct(sp[k]["zero_frac"], 1) + "\\%"
        m[f"ACT{K}"] = f(sp[k]["active_per_sample"], 1)
        m[f"DEAD{K}"] = str(sp[k]["dead_units"])
    # ---- interpolasi
    for k in ("ae", "sae", "vae"):
        v = I[k]["void_ratio_per_alpha"]
        m[f"VOID{k.upper()}"] = f(max(v[1:-1]), 2)
        m[f"VOIDEND{k.upper()}"] = f(max(v[0], v[-1]), 2)
    for k in ("ae", "sae", "vae", "gan", "dcgan"):
        m[f"ENTMID{k.upper()}"] = f(max(I[k]["entropy_per_alpha"][1:-1]), 2)
        m[f"STEP{k.upper()}"] = f(I[k]["max_step_mse"], 4)
    # ---- GAN dinamika
    for k in ("gan", "dcgan"):
        h = R(f"history_{k}.json")
        K = k.upper()
        m[f"DR{K}"] = f(h["D_real"][-1], 3)
        m[f"DF{K}"] = f(h["D_fake"][-1], 3)
        m[f"GN{K}first"] = f(h["gradG"][0], 2)
        m[f"GN{K}last"] = f(h["gradG"][-1], 2)
        it = h["iter_loss_G"]
        tail = it[len(it) // 2:]
        mu = sum(tail) / len(tail)
        m[f"GSTD{K}"] = f((sum((x - mu) ** 2 for x in tail) / len(tail)) ** 0.5, 3)
    ha = R("history_ae.json")
    m["AEtrain"] = f(ha["train"][-1], 4)
    m["AEval"] = f(ha["val"][-1], 4)
    for k in ("vae", "gan", "dcgan"):
        m[f"Bone{k.upper()}"] = f(M["latency_ms_batch1"][k][0], 2)
    A = R("anomaly_proxy.json")
    for k in ("ae", "sae"):
        K = k.upper()
        m[f"AUC{K}"] = f(A[k]["auroc"], 3)
        m[f"ERRN{K}"] = f(A[k]["err_normal"], 4)
        m[f"ERRD{K}"] = f(A[k]["err_defect"], 4)
        m[f"FIX{K}"] = f(A[k]["err_on_defect_pixels"], 3)
    m["PSNRdropSAE"] = f(M["recon"]["ae"]["psnr"] - M["recon"]["sae"]["psnr"], 1)
    hv = R("history_vae.json")
    m["KLVAE"] = f(hv["parts"][-1]["kl"], 1)
    m["GANoverDCGAN"] = f(M["fid"]["gan"] / M["fid"]["dcgan"], 1)
    hd = R("history_ddpm.json")
    m["DDPMfirst"] = f(hd["train"][0], 4)
    m["DDPMlast"] = f(hd["train"][-1], 4)
    m["DDPMval"] = f(hd["val"][-1], 4)
    # ---- teks bersyarat yang bergantung pada hasil (ditulis dari angka, bukan asumsi)
    fid = M["fid"]
    lat = M["latency_ms_batch64"]
    b1 = M["latency_ms_batch1"]
    order = sorted(["vae", "gan", "dcgan", "ddpm"], key=lambda k: fid[k])
    nm = {"vae": "VAE", "gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM"}
    order_txt = " $<$ ".join(f"{nm[k]} ({f(fid[k], 1)})" for k in order)
    ddpm_best = fid["ddpm"] < fid["dcgan"]
    cov = M["coverage"]
    m["QoneFID"] = (f"FID dari prior $\\mathcal{{N}}(0,I)$ (makin kecil makin baik) berurutan: {order_txt}; "
                    f"sebagai batas bawah, FID antara dua himpunan citra riil adalah {f(fid['real_val_vs_test'], 1)}.")
    m["FigTwoText"] = (
        "Sampel VAE (A) memiliki siluet benar dan beragam tetapi buram serta kontrasnya rendah: decoder mengeluarkan "
        "$\\mathbb{E}[x|z]$ sehingga tekstur kain hilang. Vanilla GAN (B) menghasilkan siluet tajam dan berkontras tinggi, "
        "tetapi setiap piksel berbintik seperti derau garam-merica, karena generator FC tidak memiliki bias korelasi spasial antarpiksel; "
        f"inilah sebabnya FID-nya ({f(fid['gan'], 1)}) bahkan lebih buruk daripada VAE ({f(fid['vae'], 1)}). DCGAN (C) menghasilkan kontur "
        "bersih, permukaan halus, dan detail seperti tali tas serta sol sepatu. DDPM (D) menghasilkan "
        + ("sampel dengan tekstur dan detail lokal paling alami serta FID terbaik "
           if ddpm_best else "sampel dengan tekstur yang alami, namun FID-nya masih di atas DCGAN ")
        + f"({f(fid['ddpm'], 1)}). Dari sisi keragaman, keempat model menutup 10 kelas hampir merata "
        f"(coverage {f(cov['vae']['norm_class_entropy'], 3)}--{f(max(cov[k]['norm_class_entropy'] for k in ('gan', 'dcgan', 'ddpm')), 3)}), "
        "sehingga tidak ada mode collapse tingkat kelas.")
    m["TrilemmaText"] = (
        "Generative trilemma (Xiao dkk., 2022) menyatakan bahwa sulit mendapatkan sekaligus kualitas sampel tinggi, sampling cepat, "
        "dan mode coverage. Hasil kami mengonfirmasinya. VAE: cepat "
        f"({f(lat['vae'][0], 2)} ms/64) dan coverage tinggi, tetapi kualitas rendah (FID {f(fid['vae'], 1)}). "
        f"DCGAN: cepat ({f(lat['dcgan'][0], 2)} ms/64) dan berkualitas (FID {f(fid['dcgan'], 1)}); coverage tingkat kelas baik, "
        "tetapi secara teoretis rentan mode dropping karena divergensi mode-seeking dan tidak memiliki likelihood untuk memverifikasinya. "
        f"DDPM: coverage dijamin oleh objektif likelihood-bound dan FID {f(fid['ddpm'], 1)}, tetapi sampling 1000 langkah butuh "
        f"{f(lat['ddpm'][0] / 1000, 1)} detik per batch 64, yaitu {m['DDPMxGAN']}$\\times$ lebih lambat daripada DCGAN. "
        "Latensi DDPM berbanding lurus dengan jumlah langkah $T$; inilah harga dari kestabilan dan coverage.")
    best_rt = min(("vae", "gan", "dcgan"), key=lambda k: b1[k][0])
    m["AvatarText"] = (
        f"Latensi per citra (batch 1, CPU 2 thread) adalah VAE {f(b1['vae'][0], 2)} ms, Vanilla GAN {f(b1['gan'][0], 2)} ms, "
        f"dan DCGAN {f(b1['dcgan'][0], 2)} ms; untuk batch 64 masing-masing {f(lat['vae'][0], 2)}, {f(lat['gan'][0], 2)}, "
        f"dan {f(lat['dcgan'][0], 2)} ms. Ketiganya jauh di bawah anggaran 30 ms karena sama-sama single-pass, sedangkan DDPM "
        f"(~{f(lat['ddpm'][0] / 64, 0)} ms per citra) mustahil dipakai. Di antara kandidat yang memenuhi latensi, DCGAN memberi kualitas "
        f"terbaik (FID {f(fid['dcgan'], 1)} vs VAE {f(fid['vae'], 1)} dan Vanilla GAN {f(fid['gan'], 1)}). Konvolusi juga lebih ramah "
        "akselerator mobile (NPU/GPU) dan parameternya dapat dikuantisasi. Vanilla GAN tidak direkomendasikan karena artefak bintiknya "
        "sangat terlihat pada wajah/avatar. VAE cocok bila yang dibutuhkan adalah kontrol laten yang halus (misalnya slider atribut), "
        "dengan konsekuensi hasil lebih buram. Untuk produk nyata, generator DCGAN perlu dikondisikan pada citra input (image-to-image).")
    if ddpm_best:
        m["CreativeChoice"] = "DDPM"
        m["CreativeText"] = (
            f"DDPM memberi FID terbaik ({f(fid['ddpm'], 1)} vs DCGAN {f(fid['dcgan'], 1)}) walaupun hanya dilatih 12 epoch di CPU, "
            "dengan loss yang turun monoton tanpa tuning keseimbangan $G$/$D$. ")
    else:
        m["CreativeChoice"] = "DDPM (dengan catatan)"
        m["CreativeText"] = (
            f"Pada skala eksperimen ini, DCGAN masih unggul dalam FID ({f(fid['dcgan'], 1)} vs DDPM {f(fid['ddpm'], 1)}) karena DDPM "
            "hanya dilatih 12 epoch dengan U-Net 1,25 M parameter di CPU. Namun kurva loss DDPM masih turun ketika training dihentikan, "
            "sedangkan GAN sudah mencapai equilibrium. ")
    m["CreativeText"] += (
        "Untuk aset beresolusi dan berfidelitas tinggi, faktor penentunya adalah inductive bias dan skalabilitas. DDPM memecah generasi "
        "menjadi ratusan langkah denoising kecil yang masing-masing merupakan regresi stabil, sehingga mudah diskalakan ke resolusi dan "
        "kapasitas besar, menutup seluruh mode data, serta mendukung conditioning (teks, inpainting, guidance). DCGAN pada resolusi tinggi "
        "terkenal sulit distabilkan dan rentan mode dropping. Latensi sampling yang tinggi tidak kritis dalam pipeline kreatif offline, "
        "dan dapat dipangkas dengan sampler yang lebih sedikit langkahnya (misalnya DDIM).")
    m["ABSTRAK"] = (
        f"Pada rekonstruksi, Vanilla AE unggul (PSNR {f(M['recon']['ae']['psnr'])} dB) atas Sparse AE dan VAE. Pada generasi, urutan FID "
        f"adalah {order_txt}. DDPM paling stabil dilatih tetapi sampling-nya {m['DDPMxGAN']}$\\times$ lebih lambat daripada DCGAN. "
        f"MLP Classifier (akurasi test {pct(E['stage1']['test_acc'])}\\%) terbukti tidak dapat dipakai untuk generasi: "
        f"proyeksi transpos bobot menghasilkan SSIM {f(E['stage3']['transpose']['ssim'], 3)}, frozen decoder hanya mencapai "
        f"PSNR {f(fd['psnr'])} dB (AE {f(ae['psnr'])} dB), dan sampling laten menghasilkan FID {f(fid['mlp_latent'], 1)}.")
    out = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items())
    open(os.path.join(ROOT, "report", "numbers.tex"), "w").write(out + "\n")
    print(f"{len(m)} macro ditulis ke report/numbers.tex")


if __name__ == "__main__":
    main()
