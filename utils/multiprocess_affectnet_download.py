import time
#!/usr/bin/env python
import os
from multiprocessing import Process
import argparse
import urllib.request
import requests
import contextlib

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def getfile(url,filename,timeout=45):
    with contextlib.closing(urllib.request.urlopen(url,timeout=timeout)) as fp:
        block_size = 1024 * 8
        block = fp.read(block_size)
        if block:
            with open(filename,'wb') as out_file:
                out_file.write(block)
                while True:
                    block = fp.read(block_size)
                    if not block:
                        break
                    out_file.write(block)
        else:
            raise Exception ('nonexisting file or connection error')

def fetch_dataset(url, save_name, save_path=""):
    dest = save_path + save_name
    print(dest)
    # r = requests.get(url)
    # with open(dest, 'w') as f:
    #     f.write(r.text)
    #urllib3.request.urlretrieve(url, dest)
    getfile(url, dest)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parallel download dataset")
    parser.add_argument('--save_path', type=str, default="/Users/aryaman/research/FER_datasets/affectNet/ ")
    # parser.add_argument('--divisions', type=int, default=50)
    args = parser.parse_args()
    divisions = 11
    urls = []
    save_names = []

    url_1 = """https://w0hn7a.dm.files.1drv.com/y4m_3XStsrbYP7SW1JrDB0UEiqbtBOqIDJkvPwr80i5yFvC28MAccMxwPCPzaNBGvpRPpRZFz6hp4eptWizSSzw_Lymv3bKGNZSL3ygkBem3Ulu3f_3DolAKE4wocH9ft_-M7OB0N2Psr3aQiJeCGL19rqzHhRLumF1-ii8oWQg-synTiwc_1tP-tR_HbQFtFCW2le5J6rpdko1j_50MkCEMg/Manually_Annotated.part01.rar?download&psid=1"""
    url_2 = """https://w0hn7a.dm.files.1drv.com/y4mfMQuuC94q04hMkOaEDBcvQZeWKXz-6pi1GacFXqY-HlqqMZrDpCVI-hgHLR5Km9TT4wEgvqWy0Hj_Y3ithp59N6eQh0k370TRmrYc6VMNN3_t_oIihAnhST7D2uzLNpswOrZ_1o0ZjGNyybElSchboqEzeNudZAZE0G5rHmgUnqeXCQsVqcy7zermR3306n6ZRqTw2zO9oa32_1tUdEbsQ/Manually_Annotated.part02.rar?download&psid=1"""
    url_3 = """https://w0hn7a.dm.files.1drv.com/y4mks6YWFzyv0YDqGv0Lje6v8a6vMtm6Ow0tW7h1UK4jCBmQzrrbRK1q4HqqLBksnELVmCFWC7JF8S8qrYQKgMxWDlRV9dMIwaEu3clbftod6gV_QluSgqzJ4syLtwxCaaRh3KU2tnZAS9-o_EM6O_F0GUeQUerekxKQCecXdSRXaAKuWSNGTmro_fFCCYzxSLrXcXfPBQ4GEQiyTVDGuJNGw/Manually_Annotated.part03.rar?download&psid=1"""
    url_4 = """https://w0hn7a.dm.files.1drv.com/y4maKPGmM0laBbYJl-pX4QdBfC8XsLe0nX9blC9JwHG30C74i6J6C8Szek1_gE1AgizvRpRntVu8G8J7-_cERVsD4nv3zjh44xYVL8lSO2y9wZ8mkneqW6B6aVs5UsZuVuQF-xY0aNtfdPazmDo_kB-KPV-3SpdzlTYUoEe6dN1KcoUeglM16OrZ8oyNE83wGDGsxM_HuV_al0_d9iHdctNhg/Manually_Annotated.part04.rar?download&psid=1"""
    url_5 = """https://w0hn7a.dm.files.1drv.com/y4meXYkczQQNFSInoggrkiWVDXnaj5SMe37QyQnBBedEki9dbx-d_VzHsGAXiW4XpmFe_1drUtIGHpJLFeWQQbnytwiH8aqEWaBTQOvCi3RjnoFqV_3s1EtI2yM1jDGBgIB2IeHTLvV1gKlAunUoSlLQwkimlo_2-KBPUZWVnmcHTQ6vJYMDhWcYrZHCrL5fi49EoZFOUvP2JQp2HJQRMDQew/Manually_Annotated.part05.rar?download&psid=1"""
    url_6 = """https://w0hn7a.dm.files.1drv.com/y4m_Skh1NmbC-qXwK3hncMmRv2AZv8XJGCGb2R4jT8RGkivPrr9y1t-UQcwHcHbtjp6-knT2zdOfvvurvju5y-V9NtDyJc7GozRgECcy-8XxARmxqRZKPZ9OFQSZfrlJiMkU9Q0XQmDzZk6SuxhCuqYTJRr1Bv_IOpM-joKH0ROKvvCLjPHY9cBUEyVCvk6xGDdbqwD0XGDeRqF8Y_HUt7nZw/Manually_Annotated.part06.rar?download&psid=1"""
    url_7 = """https://w0hn7a.dm.files.1drv.com/y4m6cgOmf81RoNI6_cbuuM2LiJXW3g5MGkI129kngmOg_BbdIYjyU8GxABsaxXYBIUAbnjyz3hzatcECAU5z2uasBFdr0gnJx9pxloO06Zz3ezhs9QE7VThbwYfiaS1HqpK1xsNpDzxz56onCxYs00gyZ9ZEbXmr1P2Q5_KIBZfZINsBYYKKHU8UexMGEah1nzKb25M9XXO_07rBEnZ1r5YqQ/Manually_Annotated.part07.rar?download&psid=1"""
    url_8 = """https://w0hn7a.dm.files.1drv.com/y4mGpKKdqatAGzflJXt1_1firb1z1edHF8y3DGKCweVvmFQwSftcuFP-M2qW2NaYI9kGKsahU0BbMg1L3NxX93WGFyJdf58_ZPy_XY8VxGS0lQSMivkW2m08hY4NECvFEk5qttQZlf5SHlhce6rFQj7qijyzWTQls0vRPBsycIDKBFz-Fw7YZypmtiQWWE3qFI_tR0KXsjE-VgQTQxMd1E7lg/Manually_Annotated.part08.rar?download&psid=1"""
    url_9 = """https://w0hn7a.dm.files.1drv.com/y4mV5rnpk1IiJ3APAb-Cby-xR8vTk7nkGLRmqpqcSVzSbc9V4cgVZCOU1DnzGSDkCVLOtqv1bV6DHBT0VKv-BgFd0y_SoAYAQOE6qsXDjN_Z6Q0AE26fZ0BmzTFQgacsWr6R-FXz7-vJfwguzPYTE69GaS2-IN1fuaHXFPalsrsCJAaINbHJLqDuthEInNsUMwFU_lkT0dvaDEBdyvDY3mdYg/Manually_Annotated.part09.rar?download&psid=1"""
    url_10 = """https://w0hn7a.dm.files.1drv.com/y4mQG9ybIwWcTvlKj0J9jNjRaoX1ZDyEF8uyQTKxDkL8NJMMqxt7RRPS_uxcsHfBbCFEsV2DWl_UyiLFuJcL90gJqcOSGzVVfeu0s-I0oHOPO7t0o-4PfKIBIwPXsUQDQNE9c62CY9m4iJd4DsUYt3q_gizOWs0PFmT5VkeNlK6WPIErmnnWmLBkwHZA4HdFVfT9fFq3kiQuaQHUW0qcQYVgw/Manually_Annotated.part10.rar?download&psid=1"""
    url_11 = """https://w0hn7a.dm.files.1drv.com/y4mVn--CD8fP7xQKMVReve3veOGQCB6bUZqdRwRnbs2FJARkJQNiWkPxMFVSZrOCGUgo2tqp6TM-F0IAJq5b0HgeNNUvxupbecEaI82or30FYzYyTBOZYaFDaqB5dEUcYxrNdLJgx_SBHmla5kCZeb6g1-X5IJUn7UA4Id06jWP8gIWmM4OySthfDTLxXv-yG4OO3DGCdHVYb9d8Oxubzi79A/Manually_Annotated.part11.rar?download&psid=1"""

    urls.append(url_1)
    urls.append(url_2)
    urls.append(url_3)
    urls.append(url_4)
    urls.append(url_5)
    urls.append(url_6)
    urls.append(url_7)
    urls.append(url_8)
    urls.append(url_9)
    urls.append(url_10)
    urls.append(url_11)

    for i in range(1, divisions+1):
        save_names.append("""Manually_Annotated.part{:02d}.rar""".format(i))

    processes = []
    for current_division in range(0, divisions):
        save_name = save_names[current_division]
        save_path = args.save_path
        p = Process(target=fetch_dataset, args=(urls[current_division], save_name, save_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
