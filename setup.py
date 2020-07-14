from setuptools import setup, find_packages

setup(
    name='pytorch_diffusion',
    version='0.0.1',
    url='https://github.com/pesser/pytorch_diffusion.git',
    author='Patrick Esser',
    author_email='patrick.esser@iwr.uni-heidelberg.de',
    description='PyTorch reimplementation of Diffusion Models',
    packages=find_packages(),    
    install_requires=[
        'torch',
        'numpy',
        'requests',
        'tqdm',
        'streamlit',
    ],
    scripts=[
        "scripts/pytorch_diffusion_demo",
    ]
)
