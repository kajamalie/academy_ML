# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:37:21 2020

@author: Kaja Amalie
"""
    

#%%

import lxml
from lxml import etree

def load_biggo():
    with open('notes.xml') as file:
        biggo = etree.XML(file.read())
    return etree.tostring(biggo)


    
with open('notes.xml') as file:
        biggo = etree.XML(file.read())
etree.tostring(biggo)

print(biggo.tag)
print(biggo.attrib)
print(biggo.element)

print(biggo.iter("to"))





#%%
with open('notes.xml') as file:
        biggo = etree.XML(file.read())
        
def num_notes():
    with open('notes.xml') as file:
        biggo = etree.XML(file.read())
        liste = []
        for note in biggo: 
            liste.append(note)
        return(len(liste))
    
                    

#%%
                    
def find_self_send(biggo):
    for note in biggo:
        tos = []
        froms = []
        for e in note.iter("to"):
            tos.append(e.text)
        for e in note.iter("from"):
            froms.append(e.text)
        froms = froms[0]
        for e in tos:
            if e == froms:
                return("Sent to self!")
            
    return ("No one sent to self.....")

#%%
def find_self_send(biggo):
    tos_froms = []
    for note in biggo:
        tos = []
        froms = []
        for e in note.iter("to"):
            tos.append(e.text)
        for e in note.iter("from"):
            froms.append(e.text)
        froms = froms[0]
        for e in tos:
            if e == froms:
                if e not in tos_froms:
                    tos_froms.append(e)
    return (', '.join(tos_froms))





#%%
# We are going to write a function that will translate each note element into a familiar 
#Python dictionary. For all these, it is useful (not mandatory) to write a few test cases 
#for the function in question before writing the code (test driven development; remember?). 

#a) Write a function that accepts a note and returns the sender as a string. 
#Write a function that accepts a note and returns the sender as a string.


# for alle avsendere (distin)
def note_string_all(biggo):
    new_str = []
    for note in biggo:
        for el in note.iter("from"):
            if el.text not in new_str:
                new_str.append(el.text)
    return(', '.join(new_str))

# for bare en avsender
note3 = biggo[34]
#print(etree.tostring(note))
def note_to_str(note):
    for e in note.iter("from"):
        return(e.text)
    

#%%
#Write a function that accepts a note and returns the list of recipients.

#for all: 
def note_from_all(biggo):
    new_list = []
    for note in biggo:
        for el in note.iter("to"):
            if el.text not in new_list:
                new_list.append(el.text)
    return(new_list, len(new_list) )


#for one
def note_from_list(note):
    to_list = []
    for el in note.iter("to"):
        to_list.append(el.text)
    return(to_list)






#%%
#Write a function that accepts a note and returns the subject as a list of words. 
#Ensure we do not get garbage like commas or empty strings in our output. Each word 
#should be lowercase.



## fulstending: 

def list_sub(note):
    sub = ''
    liste = [',', '.', '!', '?']
    sub2 = ''
    sub_list = []
    for e in note.iter("heading"):
        sub = sub + e.text.lower()
    for i in sub:
        if i in liste:
            pass
        else:
            sub2 = sub2 + i
    sub_list += sub2.split()
    return(sub_list)


#%%
#Write a function that accepts a note and returns the message as a list of lists. The outer 
#list should hold each sentence as an element. The inner lists represent the sentences as list 
#of words. Ensure we do not get garbage like commas or empty strings in our output. Each word 
#should be lowercase.


note2 = biggo[1]
note2 = biggo[1]
def message_list(note):
    str_message = ''
    list_list = []
    new_list = []
    for el in note.iter("body"):
        str_message = str_message + el.text.lower()
    list_list = str_message.split('. ')
    for sentence in list_list:
        if sentence == '':
            pass
        else:
            new_list.append(sentence.split(' '))
   # new_list[-1][-1][-1].replace('.', '')
    return(new_list)


def message_list(note):
    str_message = ''
    list_list = []
    new_list = []
    for el in note.iter("body"):
        str_message = str_message + el.text.lower()
    list_list = str_message.split('. ')
    for el in list_list:
        for i in el:
            i = i.replace('.', '')
            new_list.append(i)
    print(new_list)
        
    for
        if sentence == '':
            pass
        else:
            new_list.append(sentence.split(' '))
    return(new_list)




list2 = []
for e in list:
    e = e.replace('.', '')
    e = e.replace(',', '')
    list2.append(e)

#%%

#Apply the previous four functions to build a function that accepts a note and returns 
#a dictionary containing: the sender, the recipients, the subject, and the message.
# The dictionary holds these as the data structures specified in the above tasks. 


note1 = biggo[0]
def note_dic(note):
    note_dic = {
    'Sender': note_to_str(note),
    'Recipients': note_from_list(note),
    'Subject': list_sub(note),
    'Message': message_list(note)
    }
    return(note_dic)

dictionary = note_dic(note)


#%%
#Run the function you built above on the entire document to create a list of dictionaries, 
#where each dictionary represents a single note.

number_of_notes = num_notes()
dictionary = note_dic(note)

def biggo_list_dic(biggo):
    list_dic = []
    for note in biggo:
        note_dic(note)
        list_dic.append(note_dic(note))
    return(list_dic)

#%%
#Time for some analysis! My previous advice on using TDD remains. 
#Write a function to find out how many messages each sender has sent
function =  biggo_list_dic(biggo)
name = 'Felissima'


def message_sent(function, name):
    count = 0
    for note in function:
        if note['Sender'] == name:
            count = count + 1
        else:
            pass
    return (count)


#Fra snorre, denne lager en dictionary:
def count_sender(notes_document):
    cout_dict = {}
    for note in notes_documents:
        #find sender
        sender = get_sender_from_note(note)
        #Increment sender message count
        if sender in count_dict.keys():
            count_dict[sender] +=1 # = count_dict[sender] + 1
        else:
            count_dict[sender] = 1
    return count_dict




#%%
#Write a function to find out how many recipients each sender on average has for their 
#messages.
def get_number_of_recipients(note):
    tos =[]
    for element in note.iter('to'):
        tos.append(element)
    return len(tos)

def count_sender(notes_document):
    cout_dict = {}
    for note in notes_documents:
        #find sender
        sender = get_sender_from_note(note)
        #Increment sender message count
        if sender not in count_dict:
            count_dict[sender] = [not_recipients]
        else:
            count_dict[sender].append[nof_recipients]
    avg_recipient_dict = {}
    for sender, list_of_recipient_count in count_dict.items():
        avg_recipient_dict[sender] = sum(list_of_recipient_count/len(list_of_recipient_count))
    return avg_recipient_dict
            
        
        
        
        if sender in count_dict.keys():
            count_dict[sender] +=1 # = count_dict[sender] + 1
        else:
            count_dict[sender] = 1
    return count_dict





def message_sent(function, name):
    count = 0
    #for i in range (0, number_of_notes):
    for note in function:
        if note.iter['to'] == name:
            count = count + 1
        else:
            pass
    return (count)