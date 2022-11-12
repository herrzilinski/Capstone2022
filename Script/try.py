import json
import requests
from bs4 import BeautifulSoup

def lambda_handler(event, context):
    res = {}
    for i in event['payload'].split(","):
        # set url
        url = f"https://finance.yahoo.com/quote/{i}"

        # get the url page results
        response = requests.get(url)

        # try to parse Beautiful Soup
        try:
            soup = BeautifulSoup(response.text, "html.parser")

        except Exception as e:  # handle error gracefully
            res[i] = f'Status code: {response.status_code}, error message: {e}'
            print(f'Status code: {response.status_code}, error message: {e}')

        # find the price
        try:
            price = soup.find("fin-streamer", {'data-test': "qsp-price"}).text
            res[i] = price
            print(f"status code: {response.status_code}")
            print(f"{event['payload']}: {price}")

        except Exception as e:
            res[i] = f'Status code: {response.status_code}, error message: {e}'
            print(f'Status code: {response.status_code}, error message: {e}')

    return {
        'body': json.dumps(res),
    }

###
import json
import requests
from bs4 import BeautifulSoup


def lambda_handler(event, context):
    # set url
    url = f"https://finance.yahoo.com/quote/{event['payload']}"

    # get the url page results
    response = requests.get(url)

    # try to parse Beautiful Soup
    try:
        soup = BeautifulSoup(response.text, "html.parser")

    except Exception as e:  # handle error gracefully
        print(f'Here is the error message: {e}')
        return {
            'statusCode': response.status_code,
            'body': json.dumps(f'Here is the error message: {e}'),
        }  # send the error message back to the user

    # find the price
    try:
        price = soup.find("fin-streamer", {'data-test': "qsp-price"}).text
        print(f"status code: {response.status_code}")
        print(f"{event['payload']}: {price}")
        return {
            'statusCode': response.status_code,
            'body': json.dumps(f"{event['payload']}: {price}"),
        }
    except Exception as e:
        print(f'Here is the error message: {e}')
        return {
            'statusCode': response.status_code,
            'body': json.dumps(f'Here is the error message: {e}'),
        }

