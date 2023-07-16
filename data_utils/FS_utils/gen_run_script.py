import os

epsilon = [0.07, 0.08, 0.09, 0.1]
temperature = [0.1]
sigma = [0.75, 1.5, 1.75, 2, 2.5, 2.75]
freeze_proto_loss = [6000, 8000]


with open("script_params_fs_eval_tot.sh", "w") as f:


    for i in range(len(epsilon)):
        for j in range(len(temperature)):
            for k in range(len(sigma)):
                for l in range(len(freeze_proto_loss)):


                    desc_string = "swav_fs_all_ep{}_t{}_sigma{}".format(epsilon[i], temperature[j], sigma[k])
                    f.write("python data_utils/FS_utils/fs_train.py --batch_size 512 --epsilon {} --epochs 2500 --gr_lev eval --temperature {} --sigma {} --freeze_iters {}  --description {} --exp_root {} \n"\
                    .format(epsilon[i], temperature[j], sigma[k], freeze_proto_loss[l], desc_string, "/home/sateesh/swav_tcn_fs_eval_tot_only"))

