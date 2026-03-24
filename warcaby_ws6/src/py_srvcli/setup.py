from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'py_srvcli'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('py_srvcli/*.npy')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maciek128',
    maintainer_email='maciej.jedrzejewski6.stud@pw.edu.pl',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'service_img_check = py_srvcli.service_img_check:main',
            'service_move_servo = py_srvcli.service_move_servo:main',
            'client_img_check = py_srvcli.client_img_check:main',
        ],
    },

)
