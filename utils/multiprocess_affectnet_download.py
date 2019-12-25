from dataset import SiameseGoogleFer
import time
from torchvision import datasets, transforms
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from multiprocessing import Process
import argparse
import urllib.request


def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def fetch_dataset(url, save_name, save_path=""):
    dest = save_path + save_name
    urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parallel download dataset")
    parser.add_argument('--save_path', type=str, default="/Users/aryaman/research/FER_datasets/affectNet ")
    # parser.add_argument('--divisions', type=int, default=50)
    # args = parser.parse_args()
    divisions = 11
    urls = []
    save_names = []

    url_1 = """https://w0hn7a.dm.files.1drv.com/y4m4fqhW7MWa1F15MggNUm8GJ7DXkpmUXfL5JY2b24hIo0niR6QrTnzk5S8nG4kKA5FFqAHV4QMK3TJWi6XcVWvJjUkrDsMEuJreLx0OyWFm0LFGQInZjNIcYJCJVdRVElRC33RPcLygg6GGy_CmUJaq2ulBs3fVr42EDLn-sbdwTOM0nMymeRFo96n0OUkI2oxDLlVxzB57WVj2k3ry0zu9A/Manually_Annotated.part01.rar?download&psid=1"""
    url_2 = """https://w0hn7a.dm.files.1drv.com/y4mwyQxkaFWRv8G1Lbna1LDkf9mIGvbfrSJ4PZM-HRa3QDe-h5NQyWPhMdinZADC6EhF8oJsfn-i_F2j-L2kW9kMlhQtCRAUnS14FQXFMEZ8j0k-dawgThTyHDtxyB__a_B4LJ91AcYodbg1-6lc8IiFbBAVjFRDOutil5_lna0JuL4c1nFR4ggQMg51VQN19cKXgvnAC4btbDyx1LFTvUsmg/Manually_Annotated.part02.rar?download&psid=1"""
    url_3 = """https://w0hn7a.dm.files.1drv.com/y4mEDHWi0FEWEv21phtTzhNY1cZKkpcO5mKKbIGpbxNJn3MH2qVvGID4z4dBdFTWXgDn9A3aDtlq5NOA8r3Egdfqsw6v4TW3gum1pLLkJJw2GQOGYVEj-cji7z_YLIDf6mvxlWucQn1K3gnlTzNlkU8qsNATWc3pzohZ2WslOy91pDUDD4czpZhh-H0A0-rzmjlRq0mi_NhzRtdcGpj_qDwxA/Manually_Annotated.part03.rar?download&psid=1"""
    url_4 = """https://w0hn7a.dm.files.1drv.com/y4mpEw1IO03mpa_oGyndESVgmXI7b4z0YwMmScN2Ld3nXhcdKHCWXNiFU8nfiiTkIi8V1aM_ISp4zjSsoRwlFVnfHYg2BuzymXV8L7oi9EmW6Fk91brW2wbY3NM52rcc58PDVKaYuHHrcbuGOT2dXVAliYBj1g_5S0tFCY6QMjUcj2atZvAvVHrfmtNqFCOzev5qJy1mC3nLvmrhkwcAZP9qg/Manually_Annotated.part04.rar?download&psid=1"""
    url_5 = """https://w0hn7a.dm.files.1drv.com/y4mMv6j93SGW4DcuisvJw7tydANGEhXoCOrVFxmifD69y1x575ilFLtTzsQznn9765-JnPUUIkkWuhbtjHdmJt8r62GgiBAclx9NIze8ckUiVKKXOLzNeH0svIqfXGXej62M-7YrREYYI2AXW3gV0rsiqX-A4FhWfg1n72TaBAFZz03hf6E9eZmcQ7RV33aIfStcKvJEH5JKUhVd9hA-FsirQ/Manually_Annotated.part05.rar?download&psid=1"""
    url_6 = """https://w0hn7a.dm.files.1drv.com/y4mmS8RKLHNIgU_2gkiqGj5q7J9jAOI75ks1zlDxiOwYUm_JGsGttpWEhH46LldgvgoHUEClzqH7kZPPTBL9sQb_OJy2QskNxCfJE-nZDp5olroNEiiZfJaKomw8gP8ljftcA2AqhDh3r_Ob3tOR3RSZSajRrRCYu9wo4kuYa-rChqMFF9Nu-unHRtbBTf2HDggRXU7DbEKozqg9BkApVFBmg/Manually_Annotated.part06.rar?download&psid=1"""
    url_7 = """https://w0hn7a.dm.files.1drv.com/y4mxSof8Tyb_3RLKOGOlatqbBI63VHjPkFe6h8B30i_kuxHwP5D5SWDYnNzVk3mx1mffAAJpe2t1fN0CPbvGqWhQU9fKPpPf8Uy9ZE83-NiITVxCwacFtadKxW43kThmy8holALSvK-jhpyI3DsNR3HydkuF1LTE_c_23okxoljw9yC86D7VPTvq9gmrst2IPy9ebi2_CIn5SHQyEq_O8k8IQ/Manually_Annotated.part07.rar?download&psid=1"""
    url_8 = """https://w0hn7a.dm.files.1drv.com/y4m0vcStpZ-KQfRXPVtOZEa3XRMdjUcwCPo-LQkdWB5asccae5s49CplIv54IEqVIhh6eDQm1dqzR8aCfXBTeg3EmzHenedFh2FubuihlPdcTfuW88PQ-eEkLZXs-aiFXZnRpwEmpHpbtwZWMTiSVxGlfHXTjTdyuGGSi6wnThSHKBQPCBql5QZv12eZvdzckz3uqLUBeFpZxxpP_WeqsMMKw/Manually_Annotated.part08.rar?download&psid=1"""
    url_9 = """https://w0hn7a.dm.files.1drv.com/y4mRUOB2Og4MLIqbztT8kLdyC-lZ5RmaDodhCAhB0_rIH3Ttg4n2tOZhKMvIs214olJ-rvybRhvUo79PoIEYNqaQrDEjOhPuVUdN5LqQq9Zd4_2gtbp1dsc9jCKjwFx2rNX4NLXdAs84qHavaM-bq5fs558ZSW4skxzTagMQdLdRAH35aGC_9lngKey5tXH2cQN8P_UzIWAF7fjCCYe5r1O3g/Manually_Annotated.part09.rar?download&psid=1"""
    url_10 = """https://w0hn7a.dm.files.1drv.com/y4mXjnQdrJ_G9O5PaAYMJtnpfjqTxs9tXHcIP-fQl7OINfPT59M3h7d7_ubQXf_1UOVJbYoyvu9ztvLKazAqk_yh7UnB80l1ER4tVuVu8n1UHtBzjZMMgtmG9WvDcjNa8AO2vzoXhq8pXrpAzamzXdbz7hlgmOkwsyGTzOwIkdFhoKzMWZG6jV9_axIoAeJaICfeM1HFr7LkhScoPKpEXMlKQ/Manually_Annotated.part10.rar?download&psid=1"""
    url_11 = """https://w0hn7a.dm.files.1drv.com/y4mBmJTVFSUPUKVAqgVYj4WjV97zq3aWZBMu82ETgdtH6QELgY_cuDN5OKP50zR0g25bSi2jinogABIQuilizIKcCRdv6jmPM1uG0R5lIF45skxj1c2Lt2C4CRsa31utFrHRyggtN-IvVVqidks0yG6y7sBnqzxWO40XziQeYKiCMhK-oIS_0i8xfbRKQHY9sFtvlir1F965Z-HYdNubRDH2w/Manually_Annotated.part11.rar?download&psid=1"""

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
        p = Process(target=fetch_dataset, args=(urls[current_division], save_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
