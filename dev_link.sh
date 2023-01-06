sys_site=`python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
mkdir -p $sys_site
ln -s `pwd`/pycg $sys_site/pycg
