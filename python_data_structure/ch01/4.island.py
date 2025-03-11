class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right

def map():
    n1 = Node('H')
    n2 = Node("F")
    n3 = Node("S")
    n4 = Node("U")
    n5 = Node("E")
    n6 = Node("Z")
    n7 = Node("K")
    n8 = Node("N")
    n9 = Node("A")
    n10 = Node("Y")
    n11 = Node("I")
    n1.left = n2
    n1.right = n3
    n2.left = n4
    n2.right = n5
    n3.left = n6
    n3.right = n7
    n4.left = n8
    n5.left = n9
    n7.right = n10
    n9.right = n11
    return n1

def A_course(n):
    if n != None:
        print(n.name, '->', end='')
        A_course(n.left)
        A_course(n.right)

def B_course(n):
    if n != None:
        B_course(n.left)
        print(n.name, '->', end='')
        B_course(n.right)

def C_course(n):
    if n != None:
        C_course(n.left)
        C_course(n.right)
        print(n.name, '->', end='')

start = map()
print("A-course:\t", end='')
A_course(start)
print("\nB-course:\t", end='')
B_course(start)
print("\nC-course:\t", end='')
C_course(start)
print()
