from django import forms
from .models import *

class uform(forms.ModelForm):
    class Meta:
        model=user
        fields="__all__" 