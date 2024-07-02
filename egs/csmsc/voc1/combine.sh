# # copy feature files
# cp dump/train_nodev_16k/raw dump/combined -r
# cp dump/aishell3_16k/train_nodev/raw/* dump/combined/ -r
# cp dump/magichub_sg_16k/train_nodev/raw/* dump/combined/ -r

# combine wav.scp
for job in {1..16};do
    cat dump/train_nodev_16k/raw/wav.$job.scp dump/aishell3_16k/train_nodev/raw/wav.$job.scp dump/magichub_sg_16k/train_nodev/raw/wav.$job.scp > wav.$job.scp
    cp wav.$job.scp dump/combined/
done
