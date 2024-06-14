from django.urls import path
from . import views

urlpatterns=[
    path("", views.login), 
    path('login', views.login),
    path('logcode',views.logcode),
    
    path('logout/',views.logout_view),

    path('demo',views.demo),
    path('home',views.home),
    path('community/', views.community, name='community'),
    path('linear_regression_model/', views.linear_regression_model, name='linear_regression_model'),
    path('random_forest_model/', views.random_forest_model, name='random_forest_model'),
    path('ensemble_model/', views.ensemble_model, name='ensemble_model'),
    path('markovian_model/', views.markovian_model, name='markovian_model'),
    path('intraday_strategy/', views.intraday_strategy, name='intraday_strategy'),
    path('forecast/', views.forecast, name='forecast'),
    path('result/', views.result, name='result'),
    path('levels/', views.levels, name='levels'),

    path('quantative_intraday_strategy/', views.quantative_intraday_strategy, name='quantative_intraday_strategy'),

    path('strategy_result/', views.strategy_result, name='strategy_result'),
]