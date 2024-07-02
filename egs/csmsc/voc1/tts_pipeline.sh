#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=1      # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=2       # number of gpus in training
n_jobs=1      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/parallel_wavegan.v1.16k.finetuning.yaml
# conf=conf/parallel_wavegan.v1.16k.yaml
# directory path setting
download_dir=/data2/xintong/parallel_wavegan_downloads # direcotry to save downloaded files
dumpdir=/data2/xintong/parallel_wavegan_downloads/dump           # directory to dump features

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/magichub_sg_16k_csmsc_stats16k_finetuning/checkpoint-50000steps.pkl" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
# checkpoint="/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/train_nodev_16k_csmsc_parallel_wavegan.v1.16k/checkpoint-400000steps.pkl"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train_nodev_16k"
finetuning_set="magichub_sg_16k"
eval_set="magichub_sg_16k_gen/eval/gen_grad_100_acc_blk_grl"         # name of evaluation data direcotry
db_dir="/data2/xintong/magichub_sg"

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    local/data_prep_magichub_sg.sh \
        "${db_dir}" data/magichub_sg
fi

stats_ext=$(grep -q "hdf5" <(/home/xintong/local/bin/yq ".format" "${conf}") && echo "h5" || echo "npy")
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    # normalize and dump them
    pids=()
    for name in "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            parallel-wavegan-normalize \
                --config "${conf}" \
                --stats "${dumpdir}/${train_set}/stats.${stats_ext}" \
                --rootdir "${dumpdir}/${name}/raw" \
                --dumpdir "${dumpdir}/${name}/norm" \
                --verbose "${verbose}"
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished normalization."
fi

if [ -z "${tag}" ]; then
    expdir="/data2/xintong/parallel_wavegan_downloads/exp/${train_set}_csmsc_$(basename "${conf}" .yaml)"
else
    expdir="/data2/xintong/parallel_wavegan_downloads/exp/${finetuning_set}_csmsc_${tag}"
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    echo $outdir
    for name in "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/norm" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi
echo "Finished."
