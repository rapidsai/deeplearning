"""
Copyright (c) 2020, NVIDIA CORPORATION.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/bin/bash

python -u sub040_1_save_public.py
python -u sub040_2_save_private.py

for i in `seq 0 4`; 
do
    echo "seed ${i}";
	python -u sub040_3_predict.py --seed ${i} > log_sub040_3_predict.py_s${i}.txt;
done

python -u sub040_4_add_weight.py