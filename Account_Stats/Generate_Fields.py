# Created by Steven Farrugia 10-03-2019
# Code related to Neo4j Graph Database
# Requires package: neo4j
from datetime import datetime
import multiprocessing
from base64 import encode
import numpy as np
from neo4j import GraphDatabase
from Illicit_Accounts.Get_Illicit_Accounts import get_illicit_account_addresses
from tqdm import tqdm
from web3 import Web3
import string
import time
import requests
from web3.utils import *
from statistics import mean
def main_script():
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "1234"))
    web3 = Web3(Web3.IPCProvider('\\\\.\\pipe\\geth.ipc'))
    def get_computed_fields_from_neo4j(tx):
        counter = 0
        start = time.time()
        for record in tx.run("match (out)-[r:SENT]-(receiver)\
                            SET r.value = toFloat(r.value)\
                            WITH out.AccountID as ACCOUNT_ID, out=endNode(r) as IsReceiving, count(r) as count, r.value as value, r.timestamp as timeOfTransaction \
                            WITH ACCOUNT_ID, \
                                SUM(CASE WHEN IsReceiving THEN count ELSE NULL END) as Received_Transactions,\
                                SUM(CASE WHEN NOT IsReceiving THEN count ELSE NULL END) as Sent_Transactions,\
                                SUM(CASE WHEN IsReceiving THEN value/1000000000000000000 ELSE 0 END) as Total_Ether_Received,\
                                SUM(CASE WHEN NOT IsReceiving THEN value/1000000000000000000 ELSE 0 END) as Total_Ether_Sent,\
                                COLLECT(CASE WHEN IsReceiving THEN value/1000000000000000000 ELSE 0 END) as LIST_OF_ETHER_RECEIVED,\
                                COLLECT(CASE WHEN NOT IsReceiving THEN value/1000000000000000000 ELSE 0 END) as LIST_OF_ETHER_SENT,\
                                COLLECT(CASE WHEN IsReceiving THEN timeOfTransaction ELSE 0 END) as LIST_OF_TRANSACTION_TIMES \
                            UNWIND LIST_OF_ETHER_RECEIVED as ETHER_TRANSACTIONS_RECEIVED \
                            UNWIND LIST_OF_ETHER_SENT as ETHER_TRANSACTIONS_SENT \
                            UNWIND LIST_OF_TRANSACTION_TIMES as TRANSACTION_TIMESTAMP_SENT \
                            WITH ACCOUNT_ID, \
                                Received_Transactions,\
                                Sent_Transactions,\
                                Total_Ether_Received, \
                                Total_Ether_Sent,\
                                Total_Ether_Received - Total_Ether_Sent as Resultant_Transaction_Balance,\
                                CASE WHEN Total_Ether_Received = 0 THEN Total_Ether_Sent*100 Else Total_Ether_Sent/Total_Ether_Received*100 END as Sent_to_Received_Ether_ratio,\
                                CASE WHEN Received_Transactions = 0 THEN 0 ELSE Total_Ether_Received/Received_Transactions END AS Avg_received_Ether,\
                                CASE WHEN Sent_Transactions = 0 THEN 0 ELSE Total_Ether_Sent/Sent_Transactions END AS Avg_sent_Ether,\
                                LIST_OF_ETHER_RECEIVED, \
                                MIN(ETHER_TRANSACTIONS_RECEIVED) as MIN_RECEIVED_TRANSACTION,\
                                MAX(ETHER_TRANSACTIONS_RECEIVED) as MAX_RECEIVED_TRANSACTION,\
                                LIST_OF_ETHER_SENT,\
                                MIN(ETHER_TRANSACTIONS_SENT) as MIN_SENT_TRANSACTION,\
                                MAX(ETHER_TRANSACTIONS_SENT) as MAX_SENT_TRANSACTION \
                            RETURN ACCOUNT_ID,\
                                    Received_Transactions, \
                                    Sent_Transactions, \
                                    Total_Ether_Received,\
                                    Total_Ether_Sent, \
                                    Sent_to_Received_Ether_ratio,\
                                    Resultant_Transaction_Balance,\
                                    Avg_received_Ether,\
                                    Avg_sent_Ether, \
                                    MIN_RECEIVED_TRANSACTION,\
                                    MAX_RECEIVED_TRANSACTION,\
                                    MIN_SENT_TRANSACTION, \
                                    MAX_SENT_TRANSACTION"):
            counter = counter + 1
        end = time.time()
        print("Total time", end-start, " Total Number of accounts on which stats generated: ", counter)

    with driver.session() as session:
        session.read_transaction(get_computed_fields_from_neo4j)

def illegal_addresses_neo4j():
    driver = GraphDatabase.driver(uri, auth=("neo4j", "1234"))
    list_of_illicit_addresses = get_illicit_account_addresses()

    def get_computed_fields_from_neo4j(tx):
        #addressList = ['0x4cee68f36600debc7d48cd9e8aaa74b0994a968304054e797d9ba52c64bef998', '0x4d27e7b182c2a911a77549f52ce5d3b0bc75220f769f039573f1a7db99588185']
        pbar = tqdm(total=2610)

        for address in tqdm(list_of_illicit_addresses):
            pbar.update(1)
            for record in tx.run("match (n)-[:SENT]-(r) WHERE n.AccountID={accountID} return n", accountID=address):
                print("lol")

        pbar.close()
    with driver.session() as session:
        session.read_transaction(get_computed_fields_from_neo4j)


def get_():
    accounts = get_normal_account_addresses()



def get_normal_account_addresses():
    import pandas as pd
    csv_file = 'C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Data_processing/TX.csv'
    df = pd.read_csv(csv_file)
    senders = df.head(9000)
    receivers = df.tail(9000)
    #saved_column = df.column_name  # you can also use df['column_name']
    s = np.unique(senders['s'].values)
    r = np.unique(receivers['r'].values)
    good_accounts = np.concatenate((s,r), axis=0)
    unique_good_accounts = np.unique(good_accounts)
    print("Number of unique accounts: ", len(unique_good_accounts))
    return unique_good_accounts

# get new addresses which have not been retreived and saved from repository
def check_new_illicit_accounts():
    import pandas as pd
    data = pd.read_csv("test.csv")
    saved_addresses = data.loc[data['FLAG'] == 1]
    saved_addresses = saved_addresses['Address']
    up_to_date_illicit_addresses = get_illicit_account_addresses()
    difference_between_saved_and_updated_list = list(set(saved_addresses) - set(up_to_date_illicit_addresses))
    print(len(difference_between_saved_and_updated_list))
    return difference_between_saved_and_updated_list

def etherscanAPI():
    import csv
    #Addresses = get_illicit_account_addresses()
    Addresses = get_normal_account_addresses()
    #Addresses = check_new_illicit_accounts()

    #addresses = list_of_illicit_addresses
    addresses = Addresses[2934:]
    index = 1
    pbar = tqdm(total=len(addresses))
    for address in addresses:

        normal_tnxs = normal_transactions(index, address, flag=0)
        token_transfer_tnxs = token_transfer_transactions(address)
        try:
            all_tnxs = np.concatenate((normal_tnxs, token_transfer_tnxs), axis=None)
            #print(len(all_tnxs))
            with open(r'new_normal_addresses.csv', 'a', newline="") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(all_tnxs)
            index += 1
            pbar.update(1)
        except:
            continue

    pbar.close()


def account_balance(address):
    url = "https://api.etherscan.io/api?module=account&action=balance&address={address}" \
          "&tag=latest&apikey=YourApiKeyToken".format(address=address)

    r = requests.get(url=url)
    data = r.json()
    balance = 0

    if data['status'] != 0:
        balance = int(data['result']) / 1000000000000000000

    return balance


def get_total_number_of_normal_transactions(address):

    url = "http://api.etherscan.io/api?module=account&action=txlist&address={address}" \
          "&startblock=0&endblock=99999999&sort=asc&apikey=YourApiKeyToken".format(address=address)
    r = requests.get(url=url)
    data = r.json()
    num_normal_transactions = 0

    if data['status'] != 0:
        for tnx in range(len(data['result'])):
            num_normal_transactions += 1

    return num_normal_transactions



def token_transfer_transactions(address):
    URL = "http://api.etherscan.io/api?module=account&action=tokentx&address={address}" \
          "&startblock=0&endblock=999999999&sort=asc&apikey=YourApiKeyToken".format(address=address)

    r = requests.get(url=URL)
    data = r.json()
    #print(data)
    timestamp, recipients, timeDiffSent, timeDiffReceive, timeDiffContractTnx, receivedFromAddresses, receivedFromContractAddress, \
    sentToAddresses, sentToContractAddresses, sentToContracts, valueSent, valueReceived, valueSentContracts, \
    tokenReceivedName, tokenReceivedSymbol, tokenSentName, tokenSentSymbol, valueReceivedContract, sentToAddressesContract,\
    receivedFromAddressesContract, tokenSentNameContract, tokenSentSymbolContract = ([] for i in range(22))

    receivedTransactions, sentTransactions, minValReceived, tokenContractTnx, \
    maxValReceived, avgValReceived, minValSent, maxValSent, avgValSent, minValSentContract, \
    maxValSentContract, avgValSentContract = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ERC20_contract_tnx_fields = [0, 0, 0, 0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,
                                  0, 0, 0,0]
    if data['status'] != '0':
        for tokenTnx in range(len(data['result'])):
            timestamp.append(data['result'][tokenTnx]['timeStamp'][0])

            # if receiving
            if data['result'][tokenTnx]['to'] == address:
                receivedTransactions = receivedTransactions + 1
                receivedFromAddresses.append(data['result'][tokenTnx]['from'])
                receivedFromContractAddress.append(data['result'][tokenTnx]['contractAddress'])
                valueReceived.append(int(data['result'][tokenTnx]['value']) / 1000000000000000000)

                if data['result'][tokenTnx]['tokenName'] is not None:
                    tName = data['result'][tokenTnx]['tokenName']
                    tName.translate(str.maketrans('', '', string.punctuation))
                    tokenReceivedName.append(tName.encode("utf-8"))
                tokenReceivedSymbol.append(data['result'][tokenTnx]['tokenSymbol'])
                if receivedTransactions > 0:
                    timeDiffReceive.append((datetime.utcfromtimestamp(int(timestamp[tokenTnx])) - datetime.utcfromtimestamp(
                        int(timestamp[tokenTnx - 1]))).total_seconds() / 60)

            # if sending
            if data['result'][tokenTnx]['from'] == address:
                sentTransactions = sentTransactions + 1
                sentToAddresses.append(data['result'][tokenTnx]['to'])
                sentToContractAddresses.append(data['result'][tokenTnx]['contractAddress'])
                valueSent.append(int(data['result'][tokenTnx]['value']) / 1000000000000000000)
                if data['result'][tokenTnx]['tokenName'] is not None:
                    tName = data['result'][tokenTnx]['tokenName']
                    tName.translate(str.maketrans('', '', string.punctuation))
                    tokenSentName.append(tName.encode("utf-8"))

                tokenSentSymbol.append(data['result'][tokenTnx]['tokenSymbol'])
                if sentTransactions > 0:
                    timeDiffSent.append((datetime.utcfromtimestamp(int(timestamp[tokenTnx])) - datetime.utcfromtimestamp(
                        int(timestamp[tokenTnx - 1]))).total_seconds() / 60)

            # if a contract
            if data['result'][tokenTnx]['contractAddress'] == address:
                tokenContractTnx = tokenContractTnx + 1
                valueReceivedContract.append(int(data['result'][tokenTnx]['value']) / 1000000000000000000)
                sentToAddressesContract.append(data['result'][tokenTnx]['to'])
                receivedFromAddressesContract.append(data['result'][tokenTnx]['from'])
                if data['result'][tokenTnx]['tokenName'] is not None:
                    tokenSentNameContract.append((data['result'][tokenTnx]['tokenName']).encode("utf-8"))
                tokenSentSymbolContract.append(data['result'][tokenTnx]['tokenSymbol'])
                if tokenContractTnx > 0:
                    timeDiffContractTnx.append((datetime.utcfromtimestamp(int(timestamp[tokenTnx])) - datetime.utcfromtimestamp(
                        int(timestamp[tokenTnx - 1]))).total_seconds() / 60)

        totalTnx = receivedTransactions + sentTransactions + tokenContractTnx
        totalEtherRec = np.sum(valueReceived)
        totalEtherSent = np.sum(valueSent)
        totalEtherContract = np.sum(valueReceivedContract)
        uniqSentAddr, uniqRecAddr = uniq_addresses(sentToAddresses, receivedFromAddresses)
        uniqSentContAddr, uniqRecContAddr = uniq_addresses(sentToAddressesContract, receivedFromContractAddress)
        avgTimeBetweenSentTnx = avgTime(timeDiffSent)
        avgTimeBetweenRecTnx = avgTime(timeDiffReceive)
        avgTimeBetweenContractTnx = avgTime(timeDiffContractTnx)
        minValReceived, maxValReceived, avgValReceived = min_max_avg(valueReceived)
        minValSent, maxValSent, avgValSent = min_max_avg(valueSent)
        minValSentContract, maxValSentContract, avgValSentContract = min_max_avg(valueSentContracts)
        uniqSentTokenName = len(np.unique(tokenSentName))
        uniqRecTokenName = len(np.unique(tokenReceivedName))
        if len(tokenSentName) > 0:
            mostSentTokenType = most_frequent(tokenSentName)
        else:
            mostSentTokenType = "None"

        if len(tokenReceivedName) > 0:
            mostRecTokenType = most_frequent(tokenReceivedName)
        else:
            mostRecTokenType = "None"

        ERC20_contract_tnx_fields = [totalTnx, totalEtherRec, totalEtherSent, totalEtherContract, uniqSentAddr, uniqRecAddr,
                                     uniqSentContAddr, uniqRecContAddr, avgTimeBetweenSentTnx,
                                     avgTimeBetweenRecTnx, avgTimeBetweenRecTnx, avgTimeBetweenContractTnx,
                                     minValReceived, maxValReceived, avgValReceived,
                                     minValSent, maxValSent, avgValSent,
                                     minValSentContract, maxValSentContract, avgValSentContract,
                                     uniqSentTokenName, uniqRecTokenName, mostSentTokenType,
                                     mostRecTokenType]
    return ERC20_contract_tnx_fields

def normal_transactions(index, address, flag):
    URL = "https://api.etherscan.io/api?module=account&action=txlist&address={address}" \
          "&startblock=0&endblock=99999999&page=1&offset=100000&sort=asc&apikey=YourApiKeyToken".format(address=address)

    r = requests.get(url=URL)
    data = r.json()

    timestamp, recipients, timeDiffSent, timeDiffReceive, receivedFromAddresses, \
    sentToAddresses, sentToContracts, valueSent, valueReceived, valueSentContracts = ([] for i in range(10))
    receivedTransactions, sentTransactions, createdContracts, minValReceived, \
    maxValReceived, avgValReceived, minValSent, maxValSent, avgValSent, minValSentContract, \
    maxValSentContract, avgValSentContract = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    transaction_fields = [0, 0, 0 ,0, 0, 0,0,0, 0,0, 0, 0, 0, 0,
                         0, 0, 0,0, 0, 0,0, 0, 0, 0,0]

    if data['status'] != '0':
        for tnx in range(len(data['result'])):
            if data['result'][tnx]['isError'] == 1:
                pass
            timestamp.append(data['result'][tnx]['timeStamp'])
            if data['result'][tnx]['to'] == address:
                receivedTransactions = receivedTransactions + 1
                receivedFromAddresses.append(data['result'][tnx]['from'])
                valueReceived.append(int(data['result'][tnx]['value']) / 1000000000000000000)
                if receivedTransactions > 0:
                    timeDiffReceive.append((datetime.utcfromtimestamp(int(timestamp[tnx])) - datetime.utcfromtimestamp(
                        int(timestamp[tnx - 1]))).total_seconds() / 60)
            if data['result'][tnx]['from'] == address:
                sentTransactions = sentTransactions + 1
                sentToAddresses.append(data['result'][tnx]['to'])
                valueSent.append(int(data['result'][tnx]['value']) / 1000000000000000000)
                if sentTransactions > 0:
                    timeDiffSent.append((datetime.utcfromtimestamp(int(timestamp[tnx])) - datetime.utcfromtimestamp(
                        int(timestamp[tnx - 1]))).total_seconds() / 60)

            if data['result'][tnx]['contractAddress'] != '':
                createdContracts = createdContracts + 1
                sentToContracts.append(data['result'][tnx]['contractAddress'])
                valueSentContracts.append(int(data['result'][tnx]['value']) / 1000000000000000000)

        totalTnx = sentTransactions + receivedTransactions + createdContracts
        totalEtherReceived = np.sum(valueReceived)
        totalEtherSent = np.sum(valueSent)
        totalEtherSentContracts = np.sum(valueSentContracts)
        totalEtherBalance = totalEtherReceived - totalEtherSent - totalEtherSentContracts
        avgTimeBetweenSentTnx = avgTime(timeDiffSent)
        avgTimeBetweenRecTnx = avgTime(timeDiffReceive)
        numUniqSentAddress, numUniqRecAddress = uniq_addresses(sentToAddresses, receivedFromAddresses)
        minValReceived, maxValReceived, avgValReceived = min_max_avg(valueReceived)
        minValSent, maxValSent, avgValSent = min_max_avg(valueSent)
        minValSentContract, maxValSentContract, avgValSentContract = min_max_avg(valueSentContracts)
        timeDiffBetweenFirstAndLast = timeDiffFirstLast(timestamp)

        ILLICIT_OR_NORMAL_ACCOUNT_FLAG = flag

        transaction_fields = [index, address, ILLICIT_OR_NORMAL_ACCOUNT_FLAG ,avgTimeBetweenSentTnx, avgTimeBetweenRecTnx, timeDiffBetweenFirstAndLast,
                              sentTransactions,
                              receivedTransactions, createdContracts,
                              numUniqRecAddress, numUniqSentAddress,
                              minValReceived, maxValReceived, avgValReceived,
                              minValSent, maxValSent, avgValSent,
                              minValSentContract, maxValSentContract, avgValSentContract,
                              totalTnx, totalEtherSent, totalEtherReceived, totalEtherSentContracts,
                              totalEtherBalance]
    return transaction_fields


def timeDiffFirstLast(timestamp):
    timeDiff = 0
    if len(timestamp)>0:
        timeDiff = "{0:.2f}".format((datetime.utcfromtimestamp(int(timestamp[-1])) - datetime.utcfromtimestamp(
            int(timestamp[0]))).total_seconds() / 60)
    return timeDiff


def avgTime(timeDiff):
    timeDifference = 0
    if len(timeDiff) > 1:
        timeDifference =  "{0:.2f}".format(mean(timeDiff))
    return timeDifference

def min_max_avg(value_array_tnxs):
    minVal, maxVal, avgVal = 0, 0, 0
    if value_array_tnxs:
        minVal = min(value_array_tnxs)
        maxVal = max(value_array_tnxs)
        avgVal = mean(value_array_tnxs)
    return "{0:.6f}".format(minVal), "{0:.6f}".format(maxVal), "{0:.6f}".format(avgVal)

def uniq_addresses(sent_addresses, received_addresses):
    uniqSent, createdContrcts, uniqRec = 0, 0, 0
    if sent_addresses:
        uniqSent = len(np.unique(sent_addresses))

    if received_addresses:
        uniqRec = len(np.unique(received_addresses))
    return uniqSent, uniqRec

def most_frequent(List):
    return max(set(List), key = List.count)

if __name__ == '__main__':
    #main_script()
    #illegal_addresses_neo4j()
    #test_only_accounts()
    #get_normal_account_addresses()

    #RUN ETHERSCANAPI
    etherscanAPI()
    #check_new_illicit_accounts()