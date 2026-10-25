from regrounding import reproducible

# Before any test module imports torch: ATen and MKL read these when they start. The tests
# run on the CPU, which a busy GPU must not fail.
reproducible.pin()
reproducible.hide_gpus()
