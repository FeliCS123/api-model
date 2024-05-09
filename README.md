
Pour créer une image docker de l'application api titanic, dans un terminal, se placer dans le dossier du projet:

```
docker build -t titanic_api . 
```

Pour démarrer l'application dans un conteneur docker:
```
docker run -p 5000:5000 -it titanic_api
```
## Sinon:
Pour télécharger l'image docker de l'application api titanic à partir de dockerhub:
```
docker pull hdakhli/api-titanic
```

Pour démarrer l'application dans un conteneur docker:
```
docker run -p 5000:5000 -it hdakhli/api-titanic
```