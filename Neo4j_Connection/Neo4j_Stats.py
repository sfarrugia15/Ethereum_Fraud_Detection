# Created by Steven Farrugia 10-03-2019
# Code related to Neo4j Graph Database
# Requires package: neo4j
from base64 import encode

from neo4j import GraphDatabase
from Etherlists_And_Stats.Ethereum_Transactions import get_illicit_account_addresses
from tqdm import tqdm
from web3 import Web3
import time
from web3.utils import *
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "1234"))
web3 = Web3(Web3.IPCProvider('\\\\.\\pipe\\geth.ipc'))
def main_script():

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

def test():
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

def test_only_accounts():

    contract_address = '0xBFC39b6F805a9E40E77291afF27aeE3C96915BDD'
    filter = web3.eth.filter({"address": contract_address, "fromBlock": 1})
    logs = filter.get_all_entries()
    print(logs)

if __name__ == '__main__':
    main_script()
    #test()
    #test_only_accounts()