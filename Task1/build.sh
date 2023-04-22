set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk

    pip3 install -qq \
        catboost==1.0.6 \
        pytest==7.1.3 \
        scikit-image==0.19.3 \
        h5py==3.7.0 \
        hyperopt==0.2.7 \
        ipywidgets==8.0.2 \
        keras==2.10.0 \
        lightgbm==3.3.2 \
        matplotlib==3.5.3 \
        numpy==1.21.6 \
        pandas==1.3.5 \
        plotly==5.6.0 \
        scipy==1.7.3 \
        seaborn==0.12.0 \
        scikit-learn==1.0.2 \
        torch==1.12.1 \
        torchvision==0.13.1 \
        tqdm==4.64.1 \
        umap-learn==0.5.3 \
        xgboost==1.6.2 \
        pep8==1.7.1
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"