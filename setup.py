from setuptools import setup, find_packages

setup(
    name="hitmictools",
    version="0.4.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    author="Santiago Cano-Muñiz",
    author_email="santiago.cano-muniz@unibas.ch",
    url="https://github.com/phisanti/HiTMicTools",
    license="EUPL-1.2",
    entry_points={
        "console_scripts": [
            "hitmictools=HiTMicTools.main:main",
        ],
    },
    install_requires=[
        "torch",
        "monai",
        "numpy",
        "ome-types",
        "pandas",
        "scikit-image",
        "scipy",
        "pyyaml",
        "nd2",
        "tifffile",
        "jax==0.4.23",
        "jaxlib==0.4.23",
        "basicpy @ git+https://github.com/yuliu96/BaSiCPy.git",
        "jetraw-tools @ git+https://github.com/phisanti/jetraw_tools.git",
        "onnxruntime",
        "skl2onnx",
    ],
    keywords=["microscopy", "image-analysis", "deep-learning", "cell-segmentation"],
)
