ln -s /usr/bin/python3 /usr/bin/python
git clone https://github.com/tauzn-clock/scikit-image
cd scikit-image
pip install -r requirements.txt
pip install meson ninja pythran Cython
pip install .