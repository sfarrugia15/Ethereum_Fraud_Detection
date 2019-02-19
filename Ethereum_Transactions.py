# Created on 19.02.2019
# By Steven Farrugia
# Scope: Retrieve and analyze fraudulent activity on the Ethereum network using identified scams on etherscamdb
# Method of retrieval: EtherscamDB API, https://etherscamdb.info/api

import requests


"""
Scope: Get all documented scams using etherscamDB API
Method Info: Parse response as json file and retain relevant fields on which analysis shall be carried out
Total Number of Available documented scams: 6434 (19/02/2019)
"""
def get_scams():
    response = requests.get("https://etherscamdb.info/api/scams/")
    if response.status_code == 200:
        response = response.json()
        no_of_scams = len(response['result'])

        for scam in range(no_of_scams):
            scam_id = response['result'][scam]['id']
            scam_name = response['result'][scam]['name']
            scam_status = response['result'][scam]['status']
            print(scam_id, " ", scam_name, " ", scam_status)


"""
Scope: Get all documented addresses from EtherscamDB API
Method Info: Parse response as json file and retain addresses and scam IDs
Total Number of Available documented scam addresses: 1988 (19/02/2019)
"""
def get_scam_addresses():
    response = requests.get("https://etherscamdb.info/api/addresses/")
    if response.status_code == 200:
        response = response.json()

        key = response['result']
        print(key)
        for scam in response['result']:
            scam_address = scam
            scam_id = response['result'][scam]['id']
            print(scam_address, ' ', scam_id)


def main():
    get_scams()
    get_scam_addresses()


if __name__ == '__main__':
    main()
