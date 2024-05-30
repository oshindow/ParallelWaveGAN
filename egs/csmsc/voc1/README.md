1. single female training set
training set: 9800
conf/parallel_wavegan.v1.yaml
eval: 
/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/train_nodev_csmsc_parallel_wavegan.v1/wav/checkpoint-400000steps/magichub_sg/magichub_sg_A0001_S001_0_G0001_segment_0019_gen.wav

**result:**
<!-- magichub_sg 是16k音频，手动变成24k的话，16k以上也是空白 -->
有噪音，
timbre 会变，
16k以上的部分会被补全，--> finetuning
卡顿，
有很多伪谐波
谐波不清晰
- [ ] finetuning this model? 5854/5954 data for finetuning
    16k以上的部分也是空白，
    timbre 一致
    卡顿减少了
    噪音 (base model)
    频谱被截断 (base model)
- [ ] train from scratch?
- [ ] 16k v.s. 24k

- [ ] 1 speaker, timbre
- [ ] training set stats v.s. fintuning set stats