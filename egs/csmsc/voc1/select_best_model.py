def best_epoch(log):
    min_loss = 1000000
    min_steps = ''
    with open(log, 'r', encoding='utf8') as input:
        for line in input:
            if 'train/generator_loss' in line:
                # 2024-06-09 10:23:14,866 (train:542) INFO: (Steps: 369000) eval/generator_loss = 5.3682.
                line = line.strip().split(' ')
                steps = line[5]
                eval_loss = float(line[-1][:-1])
                if eval_loss < min_loss:
                    min_steps = steps
                    min_loss = eval_loss

        print(log, min_steps)

best_epoch('exp/train_nodev_16k_csmsc_parallel_wavegan.v1.16k.lowband.feat/train.log')
best_epoch('exp/train_nodev_16k_csmsc_parallel_wavegan.v1.16k.lowband/train.log')
best_epoch('exp/train_nodev_16k_csmsc_parallel_wavegan.v1.16k/train.log')