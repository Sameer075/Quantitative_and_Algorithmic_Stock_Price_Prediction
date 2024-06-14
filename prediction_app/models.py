from django.db import models

# Create your models here.
class user(models.Model):
    name=models.CharField(max_length=60)
    email=models.EmailField()
    pwd=models.CharField(max_length=8)

    class Meta:
        db_table="user"

    def __str__(self):
        return self.name