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
- [x] finetuning this model? 5854/5954 data for finetuning
    16k以上的部分也是空白，
    timbre 一致
    卡顿减少了
    噪音 (base model)
    频谱被截断 (base model)
    finetuning 的stats是 24k的，重新finetuning
- [ ] train from scratch?
- [x] 16k v.s. 24k

- [ ] 1 speaker, timbre
- [ ] training set stats v.s. fintuning set stats

2. 16k subband pqmf method
(Steps: 22000) train/generator_loss = 1.2336.
(Steps: 22000) eval/generator_loss = 5.4772. 很高 (正常)

噪声大
- [ ] batch size 32 vs batch size 6
    batch size 32 收敛太慢
    try batch size 6
    batch size 和 generator 的structure 有关吗
- [ ] istft reconstruct
    basemodel 效果提升不明显，finetuning 呢

64	1000
128	2000
192	3000
256	4000
320	5000
384	6000
448	7000
512	8000

prepare aishell-3 data

SSB18370413.wav 请 qing3 求 qiu2 依 yi1 法 fa3 撤 che4 销 xiao1 中 zhong1 卫 wei4 市 shi4 中 zhong1 级 ji2 人 ren2 民 min2 法 fa3 院 yuan4 的 de5 民 min2 事 shi4 裁 cai2 定 ding4 书 shu1