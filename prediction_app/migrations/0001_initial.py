# Generated by Django 4.2.2 on 2024-05-26 22:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="user",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=60)),
                ("email", models.EmailField(max_length=254)),
                ("pwd", models.CharField(max_length=8)),
            ],
            options={"db_table": "user",},
        ),
    ]
