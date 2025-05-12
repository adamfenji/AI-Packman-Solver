'''cluedo.py - project skeleton for a propositional reasoner
for the game of Clue.  Unimplemented portions have the comment "TO
BE IMPLEMENTED AS AN EXERCISE".  The reasoner does not include
knowledge of how many cards each player holds.
Originally by Todd Neller
Ported to Python by Dave Musicant
Adapted to course needs by Laura Brown

Copyright (C) 2008 Dave Musicant

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Information about the GNU General Public License is available online at:
  http://www.gnu.org/licenses/
To receive a copy of the GNU General Public License, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.'''

import cnf

class Cluedo:
    suspects = ['sc', 'mu', 'wh', 'gr', 'pe', 'pl']
    weapons  = ['kn', 'cs', 're', 'ro', 'pi', 'wr']
    rooms    = ['ha', 'lo', 'di', 'ki', 'ba', 'co', 'bi', 'li', 'st']
    casefile = "cf"
    hands    = suspects + [casefile]
    cards    = suspects + weapons + rooms

    """
    Return ID for player/card pair from player/card indicies
    """
    @staticmethod
    def getIdentifierFromIndicies(hand, card):
        return hand * len(Cluedo.cards) + card + 1

    """
    Return ID for player/card pair from player/card names
    """
    @staticmethod
    def getIdentifierFromNames(hand, card):
        return Cluedo.getIdentifierFromIndicies(Cluedo.hands.index(hand), Cluedo.cards.index(card))


# **************
#  Question 6 
# **************
def deal(hand, cards):
    clauses = [] #hold the CNF clauses
    for card in range(len(cards)):
        cardList = [] #hold the single clause
        #get unique identifier and append it to the clauses list
        cardList.append(Cluedo.getIdentifierFromNames(hand, cards[card])) 
        clauses.append(cardList)
    return clauses


# **************
#  Question 7 
# **************
def axiom_card_exists():
    clauses = [] #hold the CNF clauses
    for card in Cluedo.cards:
        clause = [] #hold the single clause for the card
        for hand in Cluedo.hands:
            #get unique identifier for the proposition and append to the list
            clause.append(Cluedo.getIdentifierFromNames(hand, card))
        clauses.append(clause)
    return clauses


# **************
#  Question 7 
# **************
def axiom_card_unique():
    clauses = [] #hold the CNF clauses
    for card in Cluedo.cards:
        for i in range(len(Cluedo.hands)):
            for j in range(i + 1, len(Cluedo.hands)):
                #create a clause representing that the card cannot be in two different places and append to the list
                clause = [-Cluedo.getIdentifierFromNames(Cluedo.hands[i], card), -Cluedo.getIdentifierFromNames(Cluedo.hands[j], card)]
                clauses.append(clause)
    return clauses


# **************
#  Question 7 
# **************
def axiom_casefile_exists():
    clauses = [] #hold the CNF clauses
    for category in [Cluedo.suspects, Cluedo.weapons, Cluedo.rooms]:
        clause = [] #hold the single clause for the category
        for card in category:
            #get unique identifier for the proposition and append to the list
            clause.append(Cluedo.getIdentifierFromNames(Cluedo.casefile, card))
        clauses.append(clause)
    return clauses


# **************
#  Question 7 
# **************
def axiom_casefile_unique():
    clauses = [] #hold the CNF clauses
    for category in [Cluedo.suspects, Cluedo.weapons, Cluedo.rooms]:
        for i in range(len(category)):
            for j in range(i + 1, len(category)):
                #create a clause representing that no two cards in each category are in the case file and append to the list
                clause = [-Cluedo.getIdentifierFromNames(Cluedo.casefile, category[i]), -Cluedo.getIdentifierFromNames(Cluedo.casefile, category[j])]
                clauses.append(clause)
    return clauses


# **************
#  Question 8 
# **************
def suggest(suggester, card1, card2, card3, refuter, cardShown): 
    clauses = []  #hold all the generated CNF clauses
    clause = []  #temp list to build individual clauses

    if cardShown is not None and refuter is not None: #if a card was shown by the refuter, add the clause indicating that
        clause.append(Cluedo.getIdentifierFromNames(refuter, cardShown))
        clauses.append(clause)
    
    elif cardShown is None and refuter is not None: #if no card was shown, but there was a refuter
        #append clauses indicating that the cards are not with the intermediate players
        for x in range(Cluedo.hands.index(suggester) + 1, Cluedo.hands.index(refuter)):
            clauses.append([-Cluedo.getIdentifierFromIndicies(x, Cluedo.cards.index(card1))])
            clauses.append([-Cluedo.getIdentifierFromIndicies(x, Cluedo.cards.index(card2))])
            clauses.append([-Cluedo.getIdentifierFromIndicies(x, Cluedo.cards.index(card3))])
        #append clause indicating that one of the suggested cards is with the refuter
        clause.append(Cluedo.getIdentifierFromNames(refuter, card1))
        clause.append(Cluedo.getIdentifierFromNames(refuter, card2))
        clause.append(Cluedo.getIdentifierFromNames(refuter, card3))
        clauses.append(clause)
    
    else: #if no one refuted
        for suspect in Cluedo.suspects:
            if suspect is not suggester:
                clauses.append([-Cluedo.getIdentifierFromNames(suspect, card1)])
                clauses.append([-Cluedo.getIdentifierFromNames(suspect, card2)])
                clauses.append([-Cluedo.getIdentifierFromNames(suspect, card3)])

    return clauses


# **************
#  Question 9 
# **************
def accuse(accuser, card1, card2, card3, correct):
    clauses = []  #hold all the generated CNF clauses
    clause = []  #temp list to build individual clauses

    if correct: #if the accusation is correct, the cards in the case file
        clause.append(Cluedo.getIdentifierFromNames("cf", card1))
        clauses.append(clause)
        clause.append(Cluedo.getIdentifierFromNames("cf", card2))
        clauses.append(clause)
        clause.append(Cluedo.getIdentifierFromNames("cf", card3))
        clauses.append(clause)
    
    else: #if the accusation is incorrect, the cards in the accuser's hand
        clause.append(Cluedo.getIdentifierFromNames(accuser, card1))
        clauses.append(clause)
        clause.append(Cluedo.getIdentifierFromNames(accuser, card2))
        clauses.append(clause)
        clause.append(Cluedo.getIdentifierFromNames(accuser, card3))
        clauses.append(clause)
        clause.append(-Cluedo.getIdentifierFromNames("cf", card1))
        clause.append(-Cluedo.getIdentifierFromNames("cf", card2))
        clause.append(-Cluedo.getIdentifierFromNames("cf", card3))
        clauses.append(clause)

    return clauses

