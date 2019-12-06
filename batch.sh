#!/bin/bash

# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=final
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1g 
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f19_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


# Run your program
# (">" redirects the print output of your program,
#  in this case to "output.txt")

rm -rf output.txt

./mm 400 >> output.txt
