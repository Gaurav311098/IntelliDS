from django import forms


class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(
        label="Upload Dataset",
        widget=forms.ClearableFileInput(
            attrs={
                "class": (
                    "block w-full text-sm text-gray-700 "
                    "file:mr-4 file:py-2 file:px-4 "
                    "file:rounded-lg file:border-0 "
                    "file:bg-blue-600 file:text-white "
                    "hover:file:bg-blue-700 cursor-pointer"
                ),
                "accept": ".xlsx,.xls,.csv",
            }
        ),
    )

    def clean_dataset(self):
        file = self.cleaned_data["dataset"]

        allowed = (
            ".xlsx",
            ".xls",
            ".csv",
        )

        if not file.name.lower().endswith(allowed):
            raise forms.ValidationError(
                "Only Excel (.xlsx, .xls) and CSV files are allowed."
            )

        max_size = 100 * 1024 * 1024

        if file.size > max_size:
            raise forms.ValidationError(
                "Maximum upload size is 100 MB."
            )

        return file