import os
import re
import shlex
import subprocess

from ruamel.yaml import YAML

formatter = YAML(typ="jinja2")
this_dir, this_file = os.path.split(__file__)
with open(os.path.join(this_dir, "meta.yaml")) as file_:
    data = formatter.load(file_)

# get package name
path_setup = os.path.join(os.path.dirname(this_dir), "setup.py")
package_name = (
    subprocess.run(shlex.split("python {} --name".format(path_setup)), stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

# extract dependencies
lines = set()
lines.update(data["requirements"]["run"])
lines.update(data["test"]["requires"])
if package_name in lines:
    lines.remove(package_name)

# format and write to a conda-format env file
lines = [re.sub(" ", "", line) for line in lines]
lines = [re.sub("==", "=", line) for line in lines]
lines = [line + "\n" for line in list(lines)]
dir_output = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dir_output, "environment.yml"), "w") as file_:
    file_.writelines(lines)
