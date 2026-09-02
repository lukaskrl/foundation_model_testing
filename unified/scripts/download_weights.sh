#!/usr/bin/env bash
# Fetch the pretrained encoder checkpoints for every backbone in the benchmark.
#
#   bash scripts/download_weights.sh                # all models
#   bash scripts/download_weights.sh ctfm voco      # only these keys
#   WEIGHTS_ROOT=/data/weights bash scripts/download_weights.sh
#
# Three repos are gated on HuggingFace (biomedparse, dino3d, ctclip). Accept the
# terms on the repo page once, then export a token:
#   export HF_TOKEN=hf_xxx      (or: hf auth login)
# Without a token those three are skipped and reported at the end.
#
# Resumable (curl -C -) and idempotent: a file whose size and sha256 already
# match is left alone, so re-running after an interruption costs nothing.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$REPO_ROOT/weights}"
JOBS="${JOBS:-3}"

# key|dest relative to WEIGHTS_ROOT|repo|repo path|expected bytes|sha256 (- = unknown)
# `repo` is `<owner>/<name>` for a model repo, `datasets/<owner>/<name>` for a dataset.
MANIFEST=(
"ctfm|CT-FM/ct_fm_feature_extractor/model.safetensors|project-lighter/ct_fm_feature_extractor|model.safetensors|311116176|b521ff13764ad0fce67f8ad2e5aa9ccc0823f69e4544b446581d8a2dee686215"
"ctfm|CT-FM/ct_fm_feature_extractor/config.json|project-lighter/ct_fm_feature_extractor|config.json|123|-"
"vista3d|VISTA/vista3d_pretrained_model/model_monai1.3.pt|nvidia/NV-Segment-CT|vista3d_pretrained_model/model_monai1.3.pt|872073824|889042ab37dbb9f9b2467e4a91654fb3b56482b74b98d315fc89026ab1570af8"
"voco_b|VoCo/VoComni_B.pt|Luffy503/VoCo|VoComni_B.pt|298654672|b3b3e2710eb5c8f08f0e0b54eb0791537f70efc95f725b83d8bb0389ab0dc4a6"
"voco_h|VoCo/VoComni_H.pt|Luffy503/VoCo|VoComni_H.pt|4650986176|6863af6b5a1f05a2650c867418bb25b30dc951ae63153caa317aac14632bb47a"
"suprem|SuPreM/supervised_suprem_unet_2100.pth|MrGiovanni/SuPreM|supervised_suprem_unet_2100.pth|233188229|c8dc39c86cd8d715e5ef0b5bb4ba4335ba45237127829ea016f75ee8beca0a93"
"suprem|SuPreM/supervised_suprem_swinunetr_2100.pth|MrGiovanni/SuPreM|supervised_suprem_swinunetr_2100.pth|758911849|b1672096465974502ccd5e2f2e5fce2ce214d6fb9848b23fc24f7b8888ffb4a3"
"suprem|SuPreM/supervised_suprem_segresnet_2100.pth|MrGiovanni/SuPreM|supervised_suprem_segresnet_2100.pth|56500623|2db81dc05cd9ea7234ca75e921e53e32b8716dc4cba88a6710742bfc282589a3"
"sam_med3d|SAM-Med3D/sam_med3d_turbo.pth|blueyo0/SAM-Med3D|sam_med3d_turbo.pth|402163626|899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"
# Full released CLIP checkpoint; the merlin adapter slices the image tower out of
# it at load time (keys prefixed `encode_image.i3_resnet.`).
"merlin|Merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt|stanfordmimi/Merlin|i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt|1084889630|e274ec09ba6cf86ab4e5b9923a346e1ef35b4150607896264941875975674ff6"
# --- gated: needs an HF token + accepted terms on the repo page ---
"biomedparse|BiomedParse/biomedparse_v2.ckpt|microsoft/BiomedParse|biomedparse_v2.ckpt|4460902829|-|gated"
"dino3d|3DINO/3dino_vit_weights.pth|AICONSlab/3DINO-ViT|3dino_vit_weights.pth|1417113974|-|gated"
"ctclip|CT-CLIP/models/CT-CLIP-Related/CT-CLIP_v2.pt|datasets/ibrahimhamamci/CT-RATE|models/CT-CLIP-Related/CT-CLIP_v2.pt|1769618697|-|gated"
)

TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "$TOKEN" && -r "$HOME/.cache/huggingface/token" ]]; then
    TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
fi

# An expired/revoked token is worse than none: it turns a clean "skipped, gated"
# into a 401 wall. Validate once up front and drop it if the hub rejects it.
if [[ -n "$TOKEN" ]]; then
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
            -H "Authorization: Bearer $TOKEN" https://huggingface.co/api/whoami-v2)"
    if [[ "$code" != "200" ]]; then
        echo "warning: stored HF token rejected by the hub (HTTP $code) — ignoring it."
        echo "         run 'hf auth login' or export HF_TOKEN=... to fetch the gated repos."
        TOKEN=""
    fi
fi

SKIPPED_FILE="$(mktemp)"
trap 'rm -f "$SKIPPED_FILE"' EXIT

fetch() {
    local key="$1" rel="$2" repo="$3" repo_path="$4" want_bytes="$5" want_sha="$6" gated="${7:-}"
    local dest="$WEIGHTS_ROOT/$rel"

    if [[ -n "$gated" && -z "$TOKEN" ]]; then
        echo "SKIP  $key  ($rel) — gated, no HF_TOKEN" | tee -a "$SKIPPED_FILE"
        return 0
    fi

    if [[ -f "$dest" && "$want_bytes" != "-" && "$(stat -c %s "$dest")" == "$want_bytes" ]]; then
        echo "HAVE  $key  $rel"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    if [[ -n "$HF_CLI" ]]; then
        hf_fetch "$key" "$dest" "$repo" "$repo_path" || curl_fetch "$key" "$dest" "$repo" "$repo_path" "$want_bytes"
    else
        curl_fetch "$key" "$dest" "$repo" "$repo_path" "$want_bytes"
    fi || { echo "FAIL  $key  $rel — download error" | tee -a "$SKIPPED_FILE"; return 1; }

    verify "$key" "$rel" "$dest" "$want_bytes" "$want_sha"
}

# The `hf` CLI is strongly preferred: it speaks the xet protocol (chunked and
# parallel), handles the gated-repo auth handshake, and resumes on its own. The
# plain resolve endpoint served one of these files at ~50 kB/s where hf managed
# ~85 MB/s, so this is not a marginal difference.
hf_fetch() {
    local key="$1" dest="$2" repo="$3" repo_path="$4"
    local -a extra=()
    [[ "$repo" == datasets/* ]] && { extra=(--repo-type dataset); repo="${repo#datasets/}"; }
    local stage; stage="$(mktemp -d)"
    if HF_TOKEN="$TOKEN" "$HF_CLI" download "$repo" "$repo_path" "${extra[@]}" \
            --local-dir "$stage" >/dev/null 2>&1 && [[ -f "$stage/$repo_path" ]]; then
        mv -f "$stage/$repo_path" "$dest"
        rm -rf "$stage"
        return 0
    fi
    rm -rf "$stage"
    echo "note  $key — hf transport failed, falling back to curl"
    return 1
}

curl_fetch() {
    local key="$1" dest="$2" repo="$3" repo_path="$4" want_bytes="$5"
    local url="https://huggingface.co/$repo/resolve/main/$repo_path"
    local -a auth=()
    [[ -n "$TOKEN" ]] && auth=(-H "Authorization: Bearer $TOKEN")
    if [[ -f "$dest" ]]; then
        echo "RESUME $key  ($(stat -c %s "$dest") / $want_bytes bytes)"
    fi
    if curl -fL -C - --retry 5 --retry-delay 5 --retry-connrefused \
            --progress-bar "${auth[@]}" -o "$dest" "$url"; then
        return 0
    fi
    # curl exits 33 when the server rejects a Range header on an already
    # complete file; treat a size-correct file as success anyway.
    [[ -f "$dest" && "$want_bytes" != "-" && "$(stat -c %s "$dest")" == "$want_bytes" ]]
}

verify() {
    local key="$1" rel="$2" dest="$3" want_bytes="$4" want_sha="$5"
    local got; got="$(stat -c %s "$dest")"
    if [[ "$want_bytes" != "-" && "$got" != "$want_bytes" ]]; then
        echo "FAIL  $key  $rel — size $got != expected $want_bytes" | tee -a "$SKIPPED_FILE"
        return 1
    fi
    if [[ "$want_sha" != "-" ]]; then
        if [[ "$(sha256sum "$dest" | cut -d' ' -f1)" != "$want_sha" ]]; then
            echo "FAIL  $key  $rel — sha256 mismatch" | tee -a "$SKIPPED_FILE"
            return 1
        fi
        echo "OK    $key  $rel (sha256 verified)"
    else
        echo "OK    $key  $rel ($got bytes)"
    fi
}

HF_CLI="$(command -v hf || true)"

echo "weights root: $WEIGHTS_ROOT"
[[ -n "$HF_CLI" ]] && echo "transport: hf CLI ($HF_CLI), curl fallback" \
                   || echo "transport: curl (install the 'hf' CLI for xet-accelerated transfers)"
[[ -n "$TOKEN" ]] && echo "HF token: present" || echo "HF token: none (gated repos will be skipped)"
echo

running=0
for line in "${MANIFEST[@]}"; do
    [[ "$line" == \#* ]] && continue
    IFS='|' read -r key rel repo repo_path bytes sha gated <<< "$line"
    if [[ $# -gt 0 ]]; then
        match=0
        for want in "$@"; do [[ "$key" == "$want" ]] && match=1; done
        [[ $match == 1 ]] || continue
    fi
    fetch "$key" "$rel" "$repo" "$repo_path" "$bytes" "$sha" "${gated:-}" &
    running=$((running + 1))
    if (( running >= JOBS )); then wait -n; running=$((running - 1)); fi
done
wait

# The Merlin config wants the slimmed image tower; derive it from the full
# release checkpoint once it is on disk (needs torch, so it is best-effort).
MERLIN_FULL="$WEIGHTS_ROOT/Merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
MERLIN_TOWER="$WEIGHTS_ROOT/Merlin/merlin_i3resnet_image_tower.pt"
if [[ -f "$MERLIN_FULL" && ! -f "$MERLIN_TOWER" ]]; then
    echo
    echo "slicing the Merlin image tower out of the released CLIP checkpoint..."
    (cd "$REPO_ROOT/unified" && python3 -m scripts.slim_merlin_tower) \
        || echo "warning: tower extraction failed — point model.weights at $MERLIN_FULL instead"
fi

echo
echo "==================== summary ===================="
if [[ -s "$SKIPPED_FILE" ]]; then
    cat "$SKIPPED_FILE"
    echo "------------------------------------------------"
    echo "Gated repos — accept the terms once while logged in, then re-run:"
    echo "  https://huggingface.co/microsoft/BiomedParse"
    echo "  https://huggingface.co/AICONSlab/3DINO-ViT"
    echo "  https://huggingface.co/datasets/ibrahimhamamci/CT-RATE"
    exit 1
fi
echo "all requested checkpoints present under $WEIGHTS_ROOT"
