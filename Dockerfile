FROM ultralytics/ultralytics:latest-jetson-jetpack6

# lapx fournit le module 'lap' requis par ByteTrack.
# Le wheel est fourni localement car le conteneur n'a pas accès internet.
COPY wheels/lapx-*.whl /tmp/
RUN pip install /tmp/lapx-*.whl --no-deps --quiet \
 && rm /tmp/lapx-*.whl \
 && python - <<'EOF'
# Crée un dist-info minimal pour que pkg_resources reconnaisse 'lap',
# évitant qu'Ultralytics tente un pip install réseau au démarrage.
import site, pathlib
di = pathlib.Path(site.getsitepackages()[0]) / "lap-0.9.4.dist-info"
di.mkdir(parents=True, exist_ok=True)
(di / "METADATA").write_text("Metadata-Version: 2.1\nName: lap\nVersion: 0.9.4\n")
(di / "RECORD").write_text("")
(di / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: stub\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
EOF
