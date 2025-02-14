 # Inline TODOs


# TODO

## [`zanj/loading.py`](/zanj/loading.py)

- add a separate "asserts" function?  
  local link: [`/zanj/loading.py#88`](/zanj/loading.py#88) 
  | view on GitHub: [zanj/loading.py#L88](https://github.com/mivanit/zanj/blob/main/zanj/loading.py#L88)
  | [Make Issue](https://github.com/mivanit/zanj/issues/new?title=add%20a%20separate%20%22asserts%22%20function%3F&body=%23%20source%0A%0A%5B%60zanj%2Floading.py%23L88%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fzanj%2Fblob%2Fmain%2Fzanj%2Floading.py%23L88%29%0A%0A%23%20context%0A%60%60%60python%0A%20%20%20%20%22%22%22handler%20for%20loading%20an%20object%20from%20a%20json%20file%20or%20a%20ZANJ%20archive%22%22%22%0A%0A%20%20%20%20%23%20TODO%3A%20add%20a%20separate%20%22asserts%22%20function%3F%0A%20%20%20%20%23%20right%20now%2C%20any%20asserts%20must%20happen%20in%20%60check%60%20or%20%60load%60%20which%20is%20annoying%20with%20lambdas%0A%60%60%60&labels=enhancement)

  ```python
"""handler for loading an object from a json file or a ZANJ archive"""

    # TODO: add a separate "asserts" function?
    # right now, any asserts must happen in `check` or `load` which is annoying with lambdas
  ```




## [`zanj/zanj.py`](/zanj/zanj.py)

- calling self.json_serialize again here might be slow  
  local link: [`/zanj/zanj.py#184`](/zanj/zanj.py#184) 
  | view on GitHub: [zanj/zanj.py#L184](https://github.com/mivanit/zanj/blob/main/zanj/zanj.py#L184)
  | [Make Issue](https://github.com/mivanit/zanj/issues/new?title=calling%20self.json_serialize%20again%20here%20might%20be%20slow&body=%23%20source%0A%0A%5B%60zanj%2Fzanj.py%23L184%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fzanj%2Fblob%2Fmain%2Fzanj%2Fzanj.py%23L184%29%0A%0A%23%20context%0A%60%60%60python%0A%20%20%20%20%20%20%20%20%23%20serialize%20the%20object%20--%20this%20will%20populate%20self._externals%0A%20%20%20%20%20%20%20%20%23%20TODO%3A%20calling%20self.json_serialize%20again%20here%20might%20be%20slow%0A%20%20%20%20%20%20%20%20json_data%3A%20JSONitem%20%3D%20self.json_serialize%28self.json_serialize%28obj%29%29%0A%60%60%60&labels=enhancement)

  ```python
# serialize the object -- this will populate self._externals
        # TODO: calling self.json_serialize again here might be slow
        json_data: JSONitem = self.json_serialize(self.json_serialize(obj))
  ```


- load only some part of the zanj file by passing an ObjectPath  
  local link: [`/zanj/zanj.py#226`](/zanj/zanj.py#226) 
  | view on GitHub: [zanj/zanj.py#L226](https://github.com/mivanit/zanj/blob/main/zanj/zanj.py#L226)
  | [Make Issue](https://github.com/mivanit/zanj/issues/new?title=load%20only%20some%20part%20of%20the%20zanj%20file%20by%20passing%20an%20ObjectPath&body=%23%20source%0A%0A%5B%60zanj%2Fzanj.py%23L226%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fzanj%2Fblob%2Fmain%2Fzanj%2Fzanj.py%23L226%29%0A%0A%23%20context%0A%60%60%60python%0A%20%20%20%20%29%20-%3E%20Any%3A%0A%20%20%20%20%20%20%20%20%22%22%22load%20the%20object%20from%20a%20ZANJ%20archive%0A%20%20%20%20%20%20%20%20%23%20TODO%3A%20load%20only%20some%20part%20of%20the%20zanj%20file%20by%20passing%20an%20ObjectPath%0A%20%20%20%20%20%20%20%20%22%22%22%0A%20%20%20%20%20%20%20%20file_path%20%3D%20Path%28file_path%29%0A%60%60%60&labels=enhancement)

  ```python
) -> Any:
        """load the object from a ZANJ archive
        # TODO: load only some part of the zanj file by passing an ObjectPath
        """
        file_path = Path(file_path)
  ```




