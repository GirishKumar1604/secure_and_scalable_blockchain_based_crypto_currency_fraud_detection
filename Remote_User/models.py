from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address = models.CharField(max_length=3000)
    gender = models.CharField(max_length=300)


class detect_cryptocurrency_fraud (models.Model):

    Fid= models.CharField(max_length=300)
    TxHash= models.CharField(max_length=300)
    BlockHeight= models.CharField(max_length=300)
    TimeStamp= models.CharField(max_length=300)
    Cryptocurrency= models.CharField(max_length=300)
    symbol= models.CharField(max_length=300)
    DateTime= models.CharField(max_length=300)
    Used_Entity= models.CharField(max_length=300)
    EqualDollar= models.CharField(max_length=300)
    LATITUDE= models.CharField(max_length=300)
    LONGITUDE= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



