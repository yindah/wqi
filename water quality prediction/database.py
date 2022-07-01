# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:53:01 2022

@author: USER
"""

from enum import Enum
import json

DATABASE = 'templates/database.json'

class Account(Enum):
    USER = 'User'
    ADMIN = 'Admin'

class Database:
    def __init__(self):
        with open(DATABASE) as json_file:
            self.json_object = json.load(json_file)

    def register_user(self, username:str, password:str, email:str, account:Account) -> bool:
        if not self._query_user_data(username, 'User'):
            self.json_object[Account.USER.value][username] = {
                'password': password,
                'email': email,
                'usertype': account.value
            }
            self._write_json()
            return True
        else:
            return False

    def login(self, username:str, password:str, account:Account) -> bool:
        user = self._query_user_data(username, Account.USER.value)
        if user:
            if user['password'] == password:
                return True
            else:
                return False
        else:
            return False
    
    def query_user_type(self, username:str, password:str) -> bool:
        user = self._query_user_data(username, Account.USER.value)
        if user:
            if user['password'] == password:
                if user['usertype'] == Account.ADMIN.value:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def submit_feedback(self, name:str, email:str, message:str) -> None:
        self.json_object['Feedbacks'].append(
            {
                "name": name,
                "email": email,
                "message": message
            }
        )
        self._write_json()

    def _query_user_data(self, username, account):
        return self.json_object[account].get(username, None)

    def _write_json(self) -> None:
        json_object = json.dumps(self.json_object, indent = 4)
        with open(DATABASE, "w") as json_file:
            json_file.write(json_object)