class GestionOficina:
    def __init__(self,numero,anio,descripcion,tipo):
        self.num=numero
        self.anio=anio
        self.descrip=descripcion
        self.type=tipo
    def getDatos(self):
        print(f'--Nuemero-- : {self.num}')
        print(f'--Año-- : {self.anio}')
        print(f'--Descripcion-- : {self.descrip}')
        print(f'--Tipo-- : {self.type}')
def menu():
    print('1- Crear expediente')
    print('2- Mostrar expediente')
    value=input('Ingrese un valor ')
    return value
def createExp(array):
    num=input('Ingrese su numero: ')
    anio=input('Ingrese el año: ')
    desc=input('Realize una breve descripcion del expediente: ')
    print('Tipos:')
    print('Proyecto')
    print('Nota')
    print('Condonacion')
    tipo=input('Ingrese un tipo: ')
    gestion=GestionOficina(num,anio,desc,tipo)
    array.append(gestion)
def mostrarDatos(array):
    for i in array:
        print('---------')
        i.getDatos()
        #print(i)
if __name__=='__main__':
    array=list()
    #arrayUpdate=createExp(array)
    while True:
        value=menu()
        if value=='1':      
            createExp(array)
        if value=='2':
            #array=createExp(array)
            mostrarDatos(array)
        elif value=='3':
            break
            

  
