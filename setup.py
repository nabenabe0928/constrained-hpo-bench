import os
import setuptools


requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


setuptools.setup(
    name="chpobench",
    version="0.0.3",
    author="nabenabe0928",
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url="https://github.com/nabenabe0928/constrained-hpo-bench",
    packages=["chpobench/", "chpobench/metadata/"],
    package_data={"": os.listdir("chpobench/metadata/") + ["discrete_spaces.json"]},
    python_requires=">=3.9",
    platforms=["Linux", "Darwin"],
    install_requires=requirements,
    include_package_data=True,
)
