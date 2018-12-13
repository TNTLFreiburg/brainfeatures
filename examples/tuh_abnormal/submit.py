import pandas as pd
import logging
import os
# TODO: hold_jid is not working, limiting number of cores is not working etc


# convert dictionay to cmd args in the form "--key value"
def dict_to_cmd_args(d):
    s = []
    for key, value in d.items():
        if pd.isna(key):
            continue
        s.append("--"+key+" "+str(value))
    return " ".join(s)


def main():
    # create jobs file text in this script from which a temporary file will be created
    job_file = \
        "#!/bin/bash\n" \
        "# redirect the output/error to some files\n" \
        "#$ -o /home/gemeinl/out/ -e /home/gemeinl/error/\n" \
        "export PYTHONPATH={}\n" \
        "{} {} {}\n "

    configs_file = "/home/gemeinl/code/brainfeaturedecode/examples/tuh_abnormal/configs.csv"
    # load all the configs to be run
    configs_df = pd.DataFrame.from_csv(configs_file)

    # specify python path, virtual env and python cript to be run
    python_path = '/home/gemeinl/code/brainfeaturedecode'
    virtual_env = '/home/gemeinl/venvs/auto-eeg-diag/bin/python'
    python_file = '/home/gemeinl/code/brainfeaturedecode/examples/tuh_abnormal/decode.py'

    # specify queue, temporary job file and command to submit
    # queue = "meta_core.q"
    script_name = "/home/gemeinl/jobs/sge/j_{}.pbs"
    # submit = "qsub -q meta_core.q {} {}"  # TODO: add array job flag? -t 1-300 and batch size -tc 100?
    submit = "qaad_pe {} 16 -l lr=1 {}"
    dependency = "-hold_jid {}"

    # n_parallel = 1  # number of jobs to run in parallel in a batch
    # batch_i = 0  # batch_i current batch running
    # i = 0  # i total number of run jobs

    # loop through all the configs
    for i, setting in enumerate(configs_df):
        config = configs_df[setting].to_dict()
        # if this is not the very first job, increase batch_i whenever n_parallel jobs were submitted
        # if i != 0 and i % n_parallel == 0:
        #     batch_i += 1

        cmd_args = dict_to_cmd_args(config)
        curr_job_file = job_file.format(python_path, virtual_env, python_file, cmd_args)
        curr_script_name = script_name.format(i)

        # write tmp job file and submit it to sge
        with open(curr_script_name, "w") as f:
            f.writelines(curr_job_file)

        # when this is not the first batch, add dependecy on the previous batch
        dependency_job_name = "j_" + str(i-1) + ".pbs"
        dependency_term = "" if i == 0 else dependency.format(dependency_job_name)
        curr_submit = submit.format(curr_script_name, dependency_term)
        print(curr_submit)
        os.system(curr_submit)
        #os.remove(curr_script_name)
        # i += 1


if __name__ == '__main__':
    # TODO: add arg parse for submit args as python path, virtual env etc
    logging.basicConfig(level=logging.DEBUG)
    main()
