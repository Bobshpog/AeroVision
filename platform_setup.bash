#!/bin/bash
#---------------------------------------------------------------------------------------------#
#                                       		Prep
#---------------------------------------------------------------------------------------------#
# [1] Download and install pipenv + PyCharm/other IDEs
#     * If pipenv is already installed, make sure it is in the PATH variable
# [2] For Windows: Install Git Bash
# [3] Pull the relevant code from Github
# [4] Run this script by opening up the GitBash shell in the code and running:
#     Command: bash ./platform_setup.bash
if [[ ! -f Pipfile ]]
then
  echo "Pipfile is missing"
else
  pipenv install
  pipenv shell
fi