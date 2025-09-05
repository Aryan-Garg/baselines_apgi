import os
import yaml

with open("/nobackup3/aryan/apgi/baselines/Quanta_Video_Restoration-QUIVER-/QUIVER_environment.yml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

for dependency in environment_data["dependencies"]:
    if isinstance(dependency, dict):
      for lib in dependency['pip']:
        os.system(f"pip install {lib}")
                    