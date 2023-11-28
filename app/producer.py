from kafka import KafkaProducer
import csv
import json
import time


if __name__ == '__main__':
    msgProducer = KafkaProducer(bootstrap_servers=['localhost:29092'],
                                api_version=(0, 10, 1),
                                value_serializer=lambda x: x.encode('utf-8'))

    print('Kafka Producer has been initiated')

    with open('baywheels-lite.csv') as csvFile:
        data = csv.DictReader(csvFile)
        for row in data:
            msgProducer.send('ml-classification', json.dumps(row))
            msgProducer.flush()

            print('Message sent: ' + json.dumps(row))
            time.sleep(2)

    print('Kafka message producer done')
