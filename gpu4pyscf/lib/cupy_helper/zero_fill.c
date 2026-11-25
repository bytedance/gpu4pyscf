/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
void zero_fill(void *out, size_t nbytes)
{
    const size_t block_size = 1048576;
    #pragma omp parallel for schedule(static)
    for (size_t offset = 0; offset < nbytes; offset += block_size) {
        size_t count = nbytes - offset;
        if (count > block_size) count = block_size;
        memset((char *)out + offset, 0, count);
    }
}
#ifdef __cplusplus
}
#endif
