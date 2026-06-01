#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=2     # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=2       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/parallel_wavegan.v1.16k.multiband_disc_100k.finetuning.yaml

# directory path setting
download_dir=/data2/xintong/parallel_wavegan_downloads # direcotry to save downloaded files
dumpdir=/data2/xintong/parallel_wavegan_downloads/dump           # directory to dump features

# training related setting
tag="disc_100k_finetuning"     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
# basemodel="/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/train_nodev_csmsc_parallel_wavegan.v1/checkpoint-400000steps.pkl"
basemodel="/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/train_nodev_16k_csmsc_disc_100k/checkpoint-400000steps.pkl"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

finetuning_set="magichub_sg_16k"
stats_set="train_nodev_16k"

train_set="$finetuning_set/train_nodev" # name of training data directory
dev_set="$finetuning_set/dev"           # name of development data direcotry
eval_set="$finetuning_set/eval"         # name of evaluation data direcotry

db_dir="/data2/xintong/magichub_sg"
set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    local/data_prep_magichub_sg.sh \
        "${db_dir}" data/magichub_sg_16k
fi

stats_ext=$(grep -q "hdf5" <(/home/xintong/local/bin/yq ".format" "${conf}") && echo "h5" || echo "npy")
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            parallel-wavegan-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

    # # calculate statistics for normalization
    # echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    # ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
    #     parallel-wavegan-compute-statistics \
    #         --config "${conf}" \
    #         --rootdir "${dumpdir}/${train_set}/raw" \
    #         --dumpdir "${dumpdir}/${train_set}" \
    #         --verbose "${verbose}"
    # echo "Successfully finished calculation of statistics."

    # normalize and dump them
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            parallel-wavegan-normalize \
                --config "${conf}" \
                --stats "${dumpdir}/${stats_set}/stats.${stats_ext}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm/dump.JOB" \
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
    expdir="/data2/xintong/parallel_wavegan_downloads/exp/${finetuning_set}_csmsc_$(basename "${conf}" .yaml)"
else
    expdir="/data2/xintong/parallel_wavegan_downloads/exp/${finetuning_set}_csmsc_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${stats_set}/stats.${stats_ext}" "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/norm" \
            --dev-dumpdir "${dumpdir}/${dev_set}/norm" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" \
            --pretrain "${basemodel}" 
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
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
