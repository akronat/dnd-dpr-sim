
echo "#######################"
echo "# Building for nix... #"
echo "#######################"
bash -i -c "conda activate dnd-dpr-sim && pyinstaller -F sim.py"
rm -rf build __pycache__

echo -e "\n\n"
echo "#######################"
echo "# Building for win... #"
echo "#######################"
rsync -a . /mnt/c/tmp/dnd-dpr-sim \
  --exclude build\
  --exclude dist\
  --exclude .git\
  --exclude __pycache__
(cd /mnt/c/tmp/dnd-dpr-sim && cmd.exe /c "call conda activate dnd-dpr-sim && pyinstaller -F sim.py")
rm -rf ./dist-win
cp -r /mnt/c/tmp/dnd-dpr-sim/dist/* ./dist
rm -rf /mnt/c/tmp/dnd-dpr-sim