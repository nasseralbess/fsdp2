~/fsdp2/.venv/bin/python meta_device_cpu_sharded_loading.py

ssh n2 '~/fsdp2/.venv/bin/python fsdp2/meta_device_cpu_sharded_loading.py' > n2_out.txt

ssh n4 '~/fsdp2/.venv/bin/python fsdp2/meta_device_cpu_sharded_loading.py' > n4_out.txt

ssh n5 '~/fsdp2/.venv/bin/python fsdp2/meta_device_cpu_sharded_loading.py' > n5_out.txt

