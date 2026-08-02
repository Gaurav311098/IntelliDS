from django.db import models
from django.contrib.auth.models import User

# ✅ Menu model
class Menu(models.Model):
    name = models.CharField(max_length=100, unique=True)
    url_name = models.CharField(max_length=200, unique=True, help_text="Enter Django URL name or path, e.g., 'reports' or '/reports/'")
    icon = models.CharField(max_length=100, blank=True, help_text="Optional icon class (e.g. 'fa fa-user')")
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order", "name"]

    def __str__(self):
        return self.name


# ✅ UserAccess model — connects User ↔ Menu
class UserAccess(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    menu = models.ForeignKey(Menu, on_delete=models.CASCADE)

    class Meta:
        unique_together = ("user", "menu")
        verbose_name_plural = "User Access"

    def __str__(self):
        return f"{self.user.username} → {self.menu.name}"
