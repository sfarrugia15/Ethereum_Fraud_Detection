# Created by Steven Farrugia 10-03-2019
# Code related to Neo4j Graph Database
# Requires package: neo4j
from neo4j import GraphDatabase

uri = "bolt://localhost:11013"
driver = GraphDatabase.driver(uri, auth=("test", "1234"))


def main():
    def get_incoming_and_outgoing_degree(tx):
        for record in tx.run("MATCH(a)"
                             "RETURN id(a), labels(a) as nodes,"
                             "size((a) -->()) as out, size((a) < --()) as in"):
            print(record["nodes"], " " , record["out"], " ", record["in"])

    with driver.session() as session:
        session.read_transaction(get_incoming_and_outgoing_degree)

if __name__ == '__main__':
    main()